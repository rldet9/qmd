/**
 * remote-llm.ts — backend distant (fork MIXTRIO) : embedding, reranking, expansion.
 *
 * SPEC-QMD-FORK-REMOTE-2026-001, lots 1 et 2. Chaque rôle se déclare par URI dans
 * les emplacements existants (`models.embed` / `.rerank` / `.generate` d'index.yml,
 * ou `QMD_EMBED_MODEL` / `QMD_RERANK_MODEL` / `QMD_GENERATE_MODEL`) :
 *
 *   openai:<base_url>#<model>   → POST <base_url>/embeddings          (embedding)
 *                                 POST <base_url>/rerank              (rerank, forme Cohere)
 *                                 POST <base_url>/chat/completions    (expansion)
 *   ollama:<base_url>#<model>   → POST <base_url>/api/embed           (embedding seulement)
 *   none                        → rôle DÉSACTIVÉ, aucun modèle chargé (rerank/generate)
 *
 * Règles de la spec portées ici :
 *   D-4  clé uniquement dans l'environnement (QMD_<ROLE>_API_KEY, repli QMD_API_KEY),
 *        jamais dans l'URI ; `ollama:` n'envoie aucun en-tête d'autorisation.
 *   D-5  aucun repli silencieux : une erreur est levée avec l'URL et le motif.
 *   D-6  au plus 4 requêtes en vol, ≤ 32 textes et ≤ 64 Ko par appel, timeout 30 s,
 *        3 reprises à backoff exponentiel sur 429 / 5xx / timeout, disjoncteur.
 *   D-7  les scores de rerank ne sont PAS transformés. Le serveur doit rendre des
 *        valeurs dans [0, 1], sinon erreur nommant le modèle. Mesuré sur Infinity
 *        (BAAI/bge-reranker-v2-m3) le 2026-09-04 : 0,619 pour le document pertinent
 *        contre 0,0000167 pour les autres. La sigmoïde de la PR amont #705 les
 *        écraserait à 0,65 contre 0,5 et détruirait cette discrimination.
 *   D-8  pas de tokenizer : plafond dur en caractères par entrée.
 *
 * Ce fichier n'importe que des types de llm.ts : llm.ts l'importe en valeur,
 * le graphe de modules reste acyclique au niveau des valeurs.
 */

import type {
  EmbedOptions,
  EmbeddingResult,
  GenerateOptions,
  GenerateResult,
  ModelInfo,
  Queryable,
  QueryType,
  QmdLLM,
  RerankDocument,
  RerankOptions,
  RerankResult,
} from "./llm.js";

// =============================================================================
// URI de modèle distant
// =============================================================================

export type RemoteScheme = "openai" | "ollama";

export type RemoteModelRef = {
  scheme: RemoteScheme;
  /** Base sans barre oblique finale, ex. https://gateway.example/v1 */
  baseUrl: string;
  /** Nom du modèle tel que le serveur le connaît, ex. qwen3-embedding-8k */
  model: string;
  /** L'URI complète, telle qu'écrite — c'est elle qui identifie l'espace vectoriel */
  uri: string;
};

const REMOTE_URI_RE = /^(openai|ollama):(https?:\/\/[^#\s]+)#([^\s#]+)$/i;

/** Valeur qui désactive un rôle : aucun modèle local n'est chargé, aucun appel distant. */
export const DISABLED_MODEL_URI = "none";

export function isDisabledModelUri(uri: string | undefined): boolean {
  return !!uri && uri.trim().toLowerCase() === DISABLED_MODEL_URI;
}

/** Vrai dès que l'URI porte un schéma distant, même mal formée (pour lever une erreur nommée). */
export function looksLikeRemoteModelUri(uri: string | undefined): boolean {
  return !!uri && /^(openai|ollama):/i.test(uri.trim());
}

/** Analyse une URI distante ; `null` si ce n'est pas une URI distante bien formée. */
export function parseRemoteModelUri(uri: string | undefined): RemoteModelRef | null {
  if (!uri) return null;
  const match = REMOTE_URI_RE.exec(uri.trim());
  if (!match) return null;
  return {
    scheme: match[1]!.toLowerCase() as RemoteScheme,
    baseUrl: match[2]!.replace(/\/+$/, ""),
    model: match[3]!,
    uri: uri.trim(),
  };
}

export function isRemoteModelUri(uri: string | undefined): boolean {
  return parseRemoteModelUri(uri) !== null;
}

/** Message d'erreur unique pour une URI qui ressemble à du distant sans en respecter la forme. */
export function describeRemoteUriShape(uri: string): string {
  return `URI de modèle distant invalide : ${uri}\n` +
    `Forme attendue : openai:<base_url>#<model> ou ollama:<base_url>#<model>\n` +
    `Exemples : openai:https://gateway.example/v1#qwen3-embedding-8k\n` +
    `           openai:http://reranker.internal:7997#BAAI/bge-reranker-v2-m3\n` +
    `           ollama:http://ollama.internal:11434#qwen3-embedding-8k:latest`;
}

// =============================================================================
// Clé d'API (D-4)
// =============================================================================

export type RemoteRole = "embed" | "rerank" | "generate";

/**
 * QMD_<ROLE>_API_KEY, puis QMD_API_KEY. Jamais lue dans l'URI ni dans le YAML.
 */
export function resolveRemoteApiKey(role: RemoteRole, env: NodeJS.ProcessEnv = process.env): string | undefined {
  const specific = env[`QMD_${role.toUpperCase()}_API_KEY`]?.trim();
  if (specific) return specific;
  const shared = env.QMD_API_KEY?.trim();
  return shared || undefined;
}

// =============================================================================
// Erreurs
// =============================================================================

export type RemoteErrorKind =
  | "auth"        // 401 / 403 — jamais réessayé
  | "http"        // autre 4xx — jamais réessayé
  | "retryable"   // 429 / 408 / 5xx / réseau / timeout — épuisement des reprises
  | "contract"    // réponse qui n'a pas la forme attendue (HTML, JSON incomplet…)
  | "dimension"   // dimension de vecteur différente d'un appel à l'autre
  | "score"       // score de rerank hors de [0, 1] (D-7)
  | "circuit"     // disjoncteur ouvert après échecs consécutifs
  | "unsupported" // opération non servie par ce backend
  | "invalid";    // URI ou configuration invalide

export class RemoteLLMError extends Error {
  readonly kind: RemoteErrorKind;
  readonly url: string;
  readonly status?: number;

  constructor(kind: RemoteErrorKind, url: string, message: string, status?: number) {
    super(message);
    this.name = "RemoteLLMError";
    this.kind = kind;
    this.url = url;
    this.status = status;
  }
}

/** Vrai quand le serveur a refusé la requête parce qu'elle était trop grosse. */
function isOversizedError(error: unknown): boolean {
  if (!(error instanceof RemoteLLMError)) return false;
  if (error.status === 413) return true;
  return /too large|context length|maximum context|payload|exceeds/i.test(error.message);
}

// =============================================================================
// Options et défauts (D-6, D-8)
// =============================================================================

export type RemoteClientOptions = {
  apiKey?: string;
  timeoutMs?: number;
  maxInFlight?: number;
  maxBatchItems?: number;
  maxBatchBytes?: number;
  maxRetries?: number;
  maxInputChars?: number;
  circuitFailures?: number;
  circuitOpenMs?: number;
  /** Injection pour les tests : fetch et attente. */
  fetchImpl?: typeof fetch;
  sleep?: (ms: number) => Promise<void>;
  /** Journal des avertissements (défaut : stderr). */
  warn?: (message: string) => void;
};

export const REMOTE_DEFAULTS = Object.freeze({
  timeoutMs: 30_000,
  maxInFlight: 4,
  maxBatchItems: 32,
  maxBatchBytes: 64 * 1024,
  maxRetries: 3,
  /** 3 caractères par token × 8 192 tokens de contexte (qwen3-embedding-8k). */
  maxInputChars: 3 * 8192,
  circuitFailures: 5,
  circuitOpenMs: 30_000,
});

/** Approximation sans tokenizer (D-8), la même que le découpage de store.ts. */
export const REMOTE_CHARS_PER_TOKEN = 3;

/** Plancher de troncature d'un document trop gros pour le reranker (D-7). */
const RERANK_MIN_DOC_CHARS = 32;

// =============================================================================
// Sémaphore
// =============================================================================

class Semaphore {
  private queue: (() => void)[] = [];
  private active = 0;

  constructor(private readonly limit: number) {}

  async acquire(): Promise<() => void> {
    if (this.active < this.limit) {
      this.active++;
      return () => this.release();
    }
    await new Promise<void>((resolve) => this.queue.push(resolve));
    this.active++;
    return () => this.release();
  }

  private release(): void {
    this.active = Math.max(0, this.active - 1);
    const next = this.queue.shift();
    if (next) next();
  }
}

// =============================================================================
// Transport HTTP commun (D-6) — timeout, reprises, disjoncteur, concurrence
// =============================================================================

type Resolved = Required<Omit<RemoteClientOptions, "apiKey" | "fetchImpl" | "sleep" | "warn">>;

export class RemoteHttpTransport {
  readonly ref: RemoteModelRef;
  readonly role: RemoteRole;
  readonly opts: Resolved;
  private readonly apiKey: string | undefined;
  private readonly fetchImpl: typeof fetch;
  private readonly sleep: (ms: number) => Promise<void>;
  readonly warn: (message: string) => void;
  private readonly semaphore: Semaphore;

  private consecutiveFailures = 0;
  private circuitOpenUntil = 0;
  private retryAfterMs: number | null = null;

  constructor(ref: RemoteModelRef, role: RemoteRole, options: RemoteClientOptions = {}) {
    this.ref = ref;
    this.role = role;
    // D-4 : ollama: ne porte jamais d'autorisation.
    this.apiKey = ref.scheme === "ollama" ? undefined : options.apiKey;
    this.opts = {
      timeoutMs: options.timeoutMs ?? REMOTE_DEFAULTS.timeoutMs,
      maxInFlight: Math.max(1, options.maxInFlight ?? REMOTE_DEFAULTS.maxInFlight),
      maxBatchItems: Math.max(1, options.maxBatchItems ?? REMOTE_DEFAULTS.maxBatchItems),
      maxBatchBytes: Math.max(1, options.maxBatchBytes ?? REMOTE_DEFAULTS.maxBatchBytes),
      maxRetries: Math.max(0, options.maxRetries ?? REMOTE_DEFAULTS.maxRetries),
      maxInputChars: Math.max(1, options.maxInputChars ?? REMOTE_DEFAULTS.maxInputChars),
      circuitFailures: Math.max(1, options.circuitFailures ?? REMOTE_DEFAULTS.circuitFailures),
      circuitOpenMs: Math.max(0, options.circuitOpenMs ?? REMOTE_DEFAULTS.circuitOpenMs),
    };
    this.fetchImpl = options.fetchImpl ?? fetch;
    this.sleep = options.sleep ?? ((ms) => new Promise((resolve) => setTimeout(resolve, ms)));
    this.warn = options.warn ?? ((message) => console.error(message));
    this.semaphore = new Semaphore(this.opts.maxInFlight);
  }

  get hasApiKey(): boolean {
    return !!this.apiKey;
  }

  endpointFor(path: string): string {
    return `${this.ref.baseUrl}${path}`;
  }

  /** Une requête JSON complète : concurrence bornée, reprises, disjoncteur. */
  async postJson(path: string, body: unknown): Promise<unknown> {
    const release = await this.semaphore.acquire();
    try {
      return await this.postWithRetry(this.endpointFor(path), body);
    } finally {
      release();
    }
  }

  private async postWithRetry(url: string, body: unknown): Promise<unknown> {
    let attempt = 0;
    for (;;) {
      if (Date.now() < this.circuitOpenUntil) {
        const remaining = Math.ceil((this.circuitOpenUntil - Date.now()) / 1000);
        throw new RemoteLLMError("circuit", url,
          `Disjoncteur ouvert vers ${url} après ${this.consecutiveFailures} échecs consécutifs ; nouvel essai dans ${remaining} s`);
      }
      try {
        const payload = await this.postOnce(url, body);
        this.consecutiveFailures = 0;
        return payload;
      } catch (error) {
        const remoteError = error instanceof RemoteLLMError ? error : new RemoteLLMError("retryable", url, String(error));
        const retryable = remoteError.kind === "retryable";
        if (retryable) this.noteFailure();
        if (!retryable || attempt >= this.opts.maxRetries) throw remoteError;
        attempt++;
        const backoff = this.backoffMs(attempt, remoteError);
        this.warn(`⚠ ${url} : ${remoteError.message} — reprise ${attempt}/${this.opts.maxRetries} dans ${backoff} ms`);
        await this.sleep(backoff);
      }
    }
  }

  private noteFailure(): void {
    this.consecutiveFailures++;
    if (this.consecutiveFailures >= this.opts.circuitFailures) {
      this.circuitOpenUntil = Date.now() + this.opts.circuitOpenMs;
    }
  }

  private backoffMs(attempt: number, error: RemoteLLMError): number {
    const hinted = this.retryAfterMs;
    this.retryAfterMs = null;
    if (error.status === 429 && hinted !== null) return Math.min(hinted, 10_000);
    const base = 500 * 2 ** (attempt - 1);
    const jitter = Math.floor(Math.random() * 250);
    return Math.min(base + jitter, 10_000);
  }

  private async postOnce(url: string, body: unknown): Promise<unknown> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.opts.timeoutMs);
    let response: Response;
    try {
      response = await this.fetchImpl(url, {
        method: "POST",
        headers: this.headers(),
        body: JSON.stringify(body),
        signal: controller.signal,
      });
    } catch (error) {
      const aborted = (error as { name?: string })?.name === "AbortError";
      throw new RemoteLLMError("retryable", url, aborted
        ? `timeout après ${this.opts.timeoutMs} ms`
        : `réseau : ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      clearTimeout(timer);
    }

    if (!response.ok) {
      const detail = await safeReadText(response);
      const status = response.status;
      if (status === 401 || status === 403) {
        throw new RemoteLLMError("auth", url,
          `HTTP ${status} sur ${url} pour ${this.ref.model} — clé ${this.hasApiKey ? "refusée" : `absente (QMD_${this.role.toUpperCase()}_API_KEY / QMD_API_KEY)`}${detail ? ` : ${detail}` : ""}`, status);
      }
      if (status === 429 || status === 408 || status >= 500) {
        const retryAfter = response.headers.get("retry-after");
        if (retryAfter && /^\d+$/.test(retryAfter.trim())) this.retryAfterMs = Number(retryAfter.trim()) * 1000;
        throw new RemoteLLMError("retryable", url, `HTTP ${status}${detail ? ` : ${detail}` : ""}`, status);
      }
      throw new RemoteLLMError("http", url, `HTTP ${status} sur ${url} pour ${this.ref.model}${detail ? ` : ${detail}` : ""}`, status);
    }

    const contentType = response.headers.get("content-type") ?? "";
    const raw = await safeReadText(response, 8_000_000);
    if (!/json/i.test(contentType) && !raw.trimStart().startsWith("{")) {
      // Le cas OpenWebUI : une route inconnue rend le SPA en 200 text/html.
      throw new RemoteLLMError("contract", url,
        `Réponse non JSON (${contentType || "sans content-type"}) sur ${url} : ce n'est pas la route attendue`);
    }
    try {
      return JSON.parse(raw);
    } catch {
      throw new RemoteLLMError("contract", url, `JSON illisible sur ${url}`);
    }
  }

  private headers(): Record<string, string> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      "Accept": "application/json",
    };
    if (this.apiKey) headers["Authorization"] = `Bearer ${this.apiKey}`;
    return headers;
  }
}

async function safeReadText(response: Response, limit = 600): Promise<string> {
  try {
    const text = await response.text();
    return text.length > limit ? `${text.slice(0, limit)}…` : text;
  } catch {
    return "";
  }
}

function describeError(obj: Record<string, unknown>): string {
  const err = obj.error ?? obj.detail;
  if (!err) return "";
  if (typeof err === "string") return ` : ${err}`;
  const message = (err as Record<string, unknown>).message;
  return typeof message === "string" ? ` : ${message}` : "";
}

// =============================================================================
// Client d'embeddings
// =============================================================================

export type RemoteEndpointDescription = {
  scheme: RemoteScheme;
  baseUrl: string;
  model: string;
  endpoint: string;
  hasApiKey: boolean;
  dimensions: number | null;
};

export class RemoteEmbeddingClient {
  readonly ref: RemoteModelRef;
  private readonly transport: RemoteHttpTransport;
  private dimensions: number | null = null;
  private truncationWarned = false;

  constructor(ref: RemoteModelRef, options: RemoteClientOptions = {}) {
    this.ref = ref;
    this.transport = new RemoteHttpTransport(ref, "embed", options);
  }

  get endpoint(): string {
    return this.ref.scheme === "ollama"
      ? this.transport.endpointFor("/api/embed")
      : this.transport.endpointFor("/embeddings");
  }

  get knownDimensions(): number | null {
    return this.dimensions;
  }

  describe(): RemoteEndpointDescription {
    return {
      scheme: this.ref.scheme,
      baseUrl: this.ref.baseUrl,
      model: this.ref.model,
      endpoint: this.endpoint,
      hasApiKey: this.transport.hasApiKey,
      dimensions: this.dimensions,
    };
  }

  /**
   * Vectorise des textes, dans l'ordre. Lève une RemoteLLMError au premier
   * échec définitif — jamais de `null` silencieux (D-5).
   */
  async embedBatch(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];
    const prepared = texts.map((text) => this.truncate(text));
    const batches = this.splitBatches(prepared);

    const results = new Array<number[]>(prepared.length);
    await Promise.all(batches.map(async (batch) => {
      const vectors = await this.request(batch.map((item) => item.text));
      batch.forEach((item, i) => { results[item.index] = vectors[i]!; });
    }));
    return results;
  }

  /** Sonde : un texte court, rend la dimension. Sert au démarrage et à `qmd doctor`. */
  async probe(): Promise<{ dimensions: number; latencyMs: number }> {
    const start = Date.now();
    const [vector] = await this.embedBatch(["qmd"]);
    return { dimensions: vector!.length, latencyMs: Date.now() - start };
  }

  // ---------------------------------------------------------------------------

  private truncate(text: string): string {
    const max = this.transport.opts.maxInputChars;
    if (text.length <= max) return text;
    if (!this.truncationWarned) {
      this.truncationWarned = true;
      this.transport.warn(`⚠ Texte tronqué à ${max} caractères pour ${this.ref.model} (pas de tokenizer local ; D-8)`);
    }
    return text.slice(0, max);
  }

  private splitBatches(texts: string[]): { index: number; text: string }[][] {
    const batches: { index: number; text: string }[][] = [];
    let current: { index: number; text: string }[] = [];
    let currentBytes = 0;
    texts.forEach((text, index) => {
      const bytes = Buffer.byteLength(text, "utf8");
      const wouldOverflow = current.length > 0
        && (current.length >= this.transport.opts.maxBatchItems || currentBytes + bytes > this.transport.opts.maxBatchBytes);
      if (wouldOverflow) {
        batches.push(current);
        current = [];
        currentBytes = 0;
      }
      current.push({ index, text });
      currentBytes += bytes;
    });
    if (current.length > 0) batches.push(current);
    return batches;
  }

  private async request(inputs: string[]): Promise<number[][]> {
    const path = this.ref.scheme === "ollama" ? "/api/embed" : "/embeddings";
    const body = this.ref.scheme === "ollama"
      ? { model: this.ref.model, input: inputs, truncate: true }
      : { model: this.ref.model, input: inputs, encoding_format: "float" };
    const payload = await this.transport.postJson(path, body);
    const vectors = this.parseVectors(this.endpoint, payload, inputs.length);
    this.checkDimensions(this.endpoint, vectors);
    return vectors;
  }

  private parseVectors(url: string, payload: unknown, expected: number): number[][] {
    const obj = payload as Record<string, unknown> | null;
    if (!obj || typeof obj !== "object") {
      throw new RemoteLLMError("contract", url, `Réponse inattendue sur ${url} : pas un objet JSON`);
    }
    let vectors: unknown[];
    if (this.ref.scheme === "ollama") {
      vectors = Array.isArray(obj.embeddings) ? obj.embeddings : [];
      if (vectors.length === 0) {
        throw new RemoteLLMError("contract", url, `Réponse Ollama sans champ \`embeddings\` sur ${url}${describeError(obj)}`);
      }
    } else {
      const data = Array.isArray(obj.data) ? (obj.data as Record<string, unknown>[]) : null;
      if (!data || data.length === 0) {
        throw new RemoteLLMError("contract", url, `Réponse OpenAI sans champ \`data\` sur ${url}${describeError(obj)}`);
      }
      const ordered = [...data].sort((a, b) => Number(a.index ?? 0) - Number(b.index ?? 0));
      vectors = ordered.map((item) => item.embedding);
    }
    if (vectors.length !== expected) {
      throw new RemoteLLMError("contract", url, `${vectors.length} vecteur(s) reçus pour ${expected} texte(s) sur ${url}`);
    }
    return vectors.map((vector, i) => {
      if (!Array.isArray(vector) || vector.length === 0 || !vector.every((x) => typeof x === "number" && Number.isFinite(x))) {
        throw new RemoteLLMError("contract", url, `Vecteur ${i} invalide dans la réponse de ${url}`);
      }
      return vector as number[];
    });
  }

  private checkDimensions(url: string, vectors: number[][]): void {
    for (const vector of vectors) {
      if (this.dimensions === null) {
        this.dimensions = vector.length;
      } else if (vector.length !== this.dimensions) {
        throw new RemoteLLMError("dimension", url,
          `Dimension inattendue : ${this.ref.model} sur ${url} a rendu ${vector.length} dims alors que les précédents faisaient ${this.dimensions} — deux modèles différents derrière le même nom ?`);
      }
    }
  }
}

// =============================================================================
// Client de reranking (forme Cohere) — D-7
// =============================================================================

export type RemoteRerankHit = { index: number; score: number };

export class RemoteRerankClient {
  readonly ref: RemoteModelRef;
  private readonly transport: RemoteHttpTransport;

  constructor(ref: RemoteModelRef, options: RemoteClientOptions = {}) {
    this.ref = ref;
    this.transport = new RemoteHttpTransport(ref, "rerank", options);
  }

  get endpoint(): string {
    return this.transport.endpointFor("/rerank");
  }

  describe(): RemoteEndpointDescription {
    return {
      scheme: this.ref.scheme,
      baseUrl: this.ref.baseUrl,
      model: this.ref.model,
      endpoint: this.endpoint,
      hasApiKey: this.transport.hasApiKey,
      dimensions: null,
    };
  }

  /**
   * Classe `documents` par pertinence pour `query`. Rend un score par document,
   * dans l'ordre des documents fournis. Les scores sont rendus **tels quels**
   * (D-7) ; une valeur hors de [0, 1] est une erreur, pas un signal à normaliser.
   */
  async rerank(query: string, documents: string[]): Promise<RemoteRerankHit[]> {
    if (documents.length === 0) return [];
    return this.rerankSlice(query, documents.map((text, index) => ({ index, text })));
  }

  /** Sonde de diagnostic : un document, rend le score et la latence. */
  async probe(): Promise<{ score: number; latencyMs: number }> {
    const start = Date.now();
    const [hit] = await this.rerank("qmd", ["qmd is a local search engine"]);
    return { score: hit!.score, latencyMs: Date.now() - start };
  }

  // ---------------------------------------------------------------------------

  /**
   * Un lot, avec récupération sur refus « trop gros » : bissection du lot, puis
   * troncature de moitié d'un document isolé jusqu'à un plancher.
   */
  private async rerankSlice(query: string, items: { index: number; text: string }[]): Promise<RemoteRerankHit[]> {
    try {
      const payload = await this.transport.postJson("/rerank", {
        model: this.ref.model,
        query,
        documents: items.map((item) => item.text),
        top_n: items.length,
        return_documents: false,
      });
      return this.parseHits(payload, items);
    } catch (error) {
      if (!isOversizedError(error)) throw error;

      if (items.length > 1) {
        const middle = Math.floor(items.length / 2);
        this.transport.warn(`⚠ ${this.endpoint} : lot refusé comme trop gros — bissection en ${middle} + ${items.length - middle}`);
        const [left, right] = await Promise.all([
          this.rerankSlice(query, items.slice(0, middle)),
          this.rerankSlice(query, items.slice(middle)),
        ]);
        return [...left, ...right];
      }

      const only = items[0]!;
      if (only.text.length <= RERANK_MIN_DOC_CHARS) throw error;
      const halved = only.text.slice(0, Math.max(RERANK_MIN_DOC_CHARS, Math.floor(only.text.length / 2)));
      this.transport.warn(`⚠ ${this.endpoint} : document tronqué à ${halved.length} caractères pour passer`);
      return this.rerankSlice(query, [{ index: only.index, text: halved }]);
    }
  }

  private parseHits(payload: unknown, items: { index: number; text: string }[]): RemoteRerankHit[] {
    const url = this.endpoint;
    const obj = payload as Record<string, unknown> | null;
    if (!obj || typeof obj !== "object") {
      throw new RemoteLLMError("contract", url, `Réponse inattendue sur ${url} : pas un objet JSON`);
    }
    const results = Array.isArray(obj.results) ? (obj.results as Record<string, unknown>[]) : null;
    if (!results) {
      throw new RemoteLLMError("contract", url, `Réponse de rerank sans champ \`results\` sur ${url}${describeError(obj)}`);
    }
    if (results.length !== items.length) {
      throw new RemoteLLMError("contract", url,
        `${results.length} score(s) reçus pour ${items.length} document(s) sur ${url}`);
    }

    const hits: RemoteRerankHit[] = [];
    for (const row of results) {
      const localIndex = Number(row.index);
      const score = Number(row.relevance_score ?? row.score);
      if (!Number.isInteger(localIndex) || localIndex < 0 || localIndex >= items.length) {
        throw new RemoteLLMError("contract", url, `Index ${row.index} hors bornes dans la réponse de ${url}`);
      }
      if (!Number.isFinite(score)) {
        throw new RemoteLLMError("contract", url, `Score non numérique pour l'index ${localIndex} sur ${url}`);
      }
      // D-7 : jamais de transformation. Un score hors [0, 1] signale un modèle qui
      // rend des log-odds ; le normaliser en silence fausserait le mélange RRF et
      // le seuil --min-score. On le refuse en nommant le modèle.
      if (score < 0 || score > 1) {
        throw new RemoteLLMError("score", url,
          `Score ${score} hors de [0, 1] rendu par ${this.ref.model} sur ${url} — ce modèle ne rend pas des probabilités. ` +
          `QMD ne transforme pas les scores (D-7) : utiliser un reranker qui rend des probabilités, ou le garder local.`);
      }
      hits.push({ index: items[localIndex]!.index, score });
    }
    return hits;
  }
}

// =============================================================================
// Client de génération (expansion de requête) et parseur partagé
// =============================================================================

const EXPANSION_SYSTEM_PROMPT =
  "Tu reformules une requête de recherche pour un moteur hybride. " +
  "Réponds UNIQUEMENT par trois lignes, sans préambule ni numérotation, de la forme exacte " +
  "`lex: …`, `vec: …`, `hyde: …`. " +
  "Écris dans la MÊME LANGUE que la requête et ne traduis jamais. " +
  "Recopie à l'identique les noms propres, sigles et identifiants techniques de la requête, sans les corriger. " +
  "lex = mots-clés. vec = question en langage naturel. hyde = un passage de 30 mots ressemblant à la réponse.";

/**
 * Parseur des lignes `type: texte` produites par un modèle d'expansion.
 * Extrait du parseur local de `LlamaCpp.expandQuery` pour être partagé — le
 * garde-fou `hasQueryTerm` est ce qui protège d'un modèle qui traduit ou
 * hallucine : une ligne ne partageant aucun terme avec la requête est jetée.
 */
export function parseExpansionLines(
  raw: string,
  query: string,
  includeLexical: boolean = true,
): Queryable[] {
  const queryTerms = query.toLowerCase().replace(/[^a-z0-9\s]/g, " ").split(/\s+/).filter(Boolean);
  const hasQueryTerm = (text: string): boolean => {
    if (queryTerms.length === 0) return true;
    const lower = text.toLowerCase();
    return queryTerms.some((term) => lower.includes(term));
  };

  const queryables = raw.trim().split("\n").map((line) => {
    const colonIdx = line.indexOf(":");
    if (colonIdx === -1) return null;
    const type = line.slice(0, colonIdx).trim().toLowerCase();
    if (type !== "lex" && type !== "vec" && type !== "hyde") return null;
    const text = line.slice(colonIdx + 1).trim().replace(/^["']|["']$/g, "");
    if (!text || !hasQueryTerm(text)) return null;
    return { type: type as QueryType, text };
  }).filter((q): q is Queryable => q !== null);

  const filtered = includeLexical ? queryables : queryables.filter((q) => q.type !== "lex");
  if (filtered.length > 0) return filtered;
  return defaultQueryables(query, includeLexical);
}

/** Repli sans expansion : la requête telle quelle, en lexical et en vectoriel. */
export function defaultQueryables(query: string, includeLexical: boolean = true): Queryable[] {
  const queries: Queryable[] = [];
  if (includeLexical) queries.push({ type: "lex", text: query });
  queries.push({ type: "vec", text: query });
  return queries;
}

export class RemoteChatClient {
  readonly ref: RemoteModelRef;
  private readonly transport: RemoteHttpTransport;

  constructor(ref: RemoteModelRef, options: RemoteClientOptions = {}) {
    this.ref = ref;
    this.transport = new RemoteHttpTransport(ref, "generate", options);
  }

  get endpoint(): string {
    return this.transport.endpointFor("/chat/completions");
  }

  describe(): RemoteEndpointDescription {
    return {
      scheme: this.ref.scheme,
      baseUrl: this.ref.baseUrl,
      model: this.ref.model,
      endpoint: this.endpoint,
      hasApiKey: this.transport.hasApiKey,
      dimensions: null,
    };
  }

  async complete(system: string, user: string, maxTokens = 300): Promise<string> {
    const payload = await this.transport.postJson("/chat/completions", {
      model: this.ref.model,
      temperature: 0.2,
      max_tokens: maxTokens,
      messages: [
        { role: "system", content: system },
        { role: "user", content: user },
      ],
    }) as Record<string, any>;

    const content = payload?.choices?.[0]?.message?.content;
    if (typeof content !== "string") {
      throw new RemoteLLMError("contract", this.endpoint,
        `Réponse de chat sans \`choices[0].message.content\` sur ${this.endpoint}${describeError(payload ?? {})}`);
    }
    return content;
  }

  /**
   * Expansion de requête. Un échec ne casse jamais la recherche : on retombe sur
   * la requête telle quelle, en journalisant — contrairement à l'embedding (D-5),
   * une expansion absente dégrade le rappel mais ne corrompt aucun index.
   */
  async expandQuery(query: string, includeLexical: boolean = true): Promise<Queryable[]> {
    try {
      const raw = await this.complete(EXPANSION_SYSTEM_PROMPT, query);
      return parseExpansionLines(raw, query, includeLexical);
    } catch (error) {
      this.transport.warn(`⚠ Expansion distante indisponible (${error instanceof Error ? error.message : String(error)}) — requête utilisée telle quelle`);
      return defaultQueryables(query, includeLexical);
    }
  }
}

// =============================================================================
// RemoteLLM — le backend vu par store.ts
// =============================================================================

export type RemoteLLMConfig = {
  /** URI complète d'embedding, ex. openai:https://…/v1#qwen3-embedding-8k */
  embedModel: string;
  /** URI de rerank, ou undefined quand le rerank n'est pas distant. */
  rerankModel?: string;
  /** URI de génération, ou undefined quand l'expansion n'est pas distante. */
  generateModel?: string;
  apiKey?: string;
  client?: RemoteClientOptions;
};

/**
 * Backend entièrement distant. Les rôles non configurés lèvent `unsupported` :
 * c'est `HybridLLM` (llm.ts) qui décide de router vers le local ou de dégrader.
 */
export class RemoteLLM implements QmdLLM {
  readonly isRemote = true;
  readonly client: RemoteEmbeddingClient;
  readonly rerankClient: RemoteRerankClient | null;
  readonly chatClient: RemoteChatClient | null;
  private readonly uri: string;
  private readonly rerankUri: string;
  private readonly generateUri: string;

  constructor(config: RemoteLLMConfig) {
    const ref = parseRemoteModelUri(config.embedModel);
    if (!ref) {
      throw new RemoteLLMError("invalid", config.embedModel, describeRemoteUriShape(config.embedModel));
    }
    this.uri = ref.uri;
    this.client = new RemoteEmbeddingClient(ref, {
      ...config.client,
      apiKey: config.apiKey ?? config.client?.apiKey ?? resolveRemoteApiKey("embed"),
    });

    const rerankRef = parseRemoteModelUri(config.rerankModel);
    this.rerankUri = rerankRef?.uri ?? "";
    this.rerankClient = rerankRef
      ? new RemoteRerankClient(rerankRef, { ...config.client, apiKey: config.client?.apiKey ?? resolveRemoteApiKey("rerank") })
      : null;

    const chatRef = parseRemoteModelUri(config.generateModel);
    this.generateUri = chatRef?.uri ?? "";
    this.chatClient = chatRef
      ? new RemoteChatClient(chatRef, { ...config.client, apiKey: config.client?.apiKey ?? resolveRemoteApiKey("generate") })
      : null;
  }

  get embedModelName(): string {
    return this.uri;
  }

  get generateModelName(): string {
    return this.generateUri;
  }

  get rerankModelName(): string {
    return this.rerankUri;
  }

  get supportsRerank(): boolean {
    return this.rerankClient !== null;
  }

  get supportsExpand(): boolean {
    return this.chatClient !== null;
  }

  describe(): RemoteEndpointDescription {
    return this.client.describe();
  }

  async embed(text: string, options: EmbedOptions = {}): Promise<EmbeddingResult | null> {
    const [vector] = await this.client.embedBatch([text]);
    return { embedding: vector!, model: options.model ?? this.uri };
  }

  async embedBatch(texts: string[], options: EmbedOptions = {}): Promise<(EmbeddingResult | null)[]> {
    const vectors = await this.client.embedBatch(texts);
    return vectors.map((embedding) => ({ embedding, model: options.model ?? this.uri }));
  }

  /** Pas de tokenizer : approximation en caractères (D-8). */
  async countTokens(text: string): Promise<number> {
    return Math.ceil(text.length / REMOTE_CHARS_PER_TOKEN);
  }

  async generate(_prompt: string, _options?: GenerateOptions): Promise<GenerateResult | null> {
    throw new RemoteLLMError("unsupported", this.client.endpoint, "generate libre n'est pas servi par le backend distant");
  }

  async modelExists(model: string): Promise<ModelInfo> {
    return { name: model, exists: true, path: this.client.endpoint };
  }

  async expandQuery(query: string, options?: { context?: string; includeLexical?: boolean }): Promise<Queryable[]> {
    const includeLexical = options?.includeLexical !== false;
    if (!this.chatClient) return defaultQueryables(query, includeLexical);
    return this.chatClient.expandQuery(query, includeLexical);
  }

  async rerank(query: string, documents: RerankDocument[], options?: RerankOptions): Promise<RerankResult> {
    if (!this.rerankClient) {
      throw new RemoteLLMError("unsupported", this.client.endpoint, "aucun modèle de rerank distant configuré");
    }
    if (documents.length === 0) return { results: [], model: options?.model ?? this.rerankUri };
    const hits = await this.rerankClient.rerank(query, documents.map((d) => d.text));
    const results = hits
      .map((hit) => ({ file: documents[hit.index]!.file, score: hit.score, index: hit.index }))
      .sort((a, b) => b.score - a.score);
    return { results, model: options?.model ?? this.rerankUri };
  }

  async dispose(): Promise<void> {
    // Rien à libérer : pas de contexte natif.
  }
}
