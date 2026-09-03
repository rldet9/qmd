/**
 * remote-llm.ts — backend d'embedding distant (fork MIXTRIO).
 *
 * SPEC-QMD-FORK-REMOTE-2026-001, lot 1. Le modèle se déclare par URI dans les
 * emplacements existants (`models.embed` d'index.yml ou `QMD_EMBED_MODEL`) :
 *
 *   openai:<base_url>#<model>   → POST <base_url>/embeddings   (OpenAI-compatible :
 *                                 LiteLLM, Ollama /v1, OpenAI, vLLM, TEI…)
 *   ollama:<base_url>#<model>   → POST <base_url>/api/embed    (API native Ollama)
 *
 * Règles de la spec portées ici :
 *   D-4  clé uniquement dans l'environnement (QMD_EMBED_API_KEY, repli QMD_API_KEY),
 *        jamais dans l'URI ; `ollama:` n'envoie aucun en-tête d'autorisation.
 *   D-5  aucun repli silencieux : une erreur est levée avec l'URL et le motif.
 *   D-6  au plus 4 requêtes en vol, ≤ 32 textes et ≤ 64 Ko par appel, timeout 30 s,
 *        3 reprises à backoff exponentiel sur 429 / 5xx / timeout, disjoncteur.
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
// Client HTTP d'embeddings
// =============================================================================

type Resolved = Required<Omit<RemoteClientOptions, "apiKey" | "fetchImpl" | "sleep" | "warn">>;

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
  private readonly apiKey: string | undefined;
  private readonly opts: Resolved;
  private readonly fetchImpl: typeof fetch;
  private readonly sleep: (ms: number) => Promise<void>;
  private readonly warn: (message: string) => void;
  private readonly semaphore: Semaphore;

  private dimensions: number | null = null;
  private consecutiveFailures = 0;
  private circuitOpenUntil = 0;
  private truncationWarned = false;

  constructor(ref: RemoteModelRef, options: RemoteClientOptions = {}) {
    this.ref = ref;
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

  get endpoint(): string {
    return this.ref.scheme === "ollama"
      ? `${this.ref.baseUrl}/api/embed`
      : `${this.ref.baseUrl}/embeddings`;
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
      hasApiKey: !!this.apiKey,
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
      const release = await this.semaphore.acquire();
      try {
        const vectors = await this.requestWithRetry(batch.map((item) => item.text));
        batch.forEach((item, i) => { results[item.index] = vectors[i]!; });
      } finally {
        release();
      }
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
    if (text.length <= this.opts.maxInputChars) return text;
    if (!this.truncationWarned) {
      this.truncationWarned = true;
      this.warn(`⚠ Texte tronqué à ${this.opts.maxInputChars} caractères pour ${this.ref.model} (pas de tokenizer local ; D-8)`);
    }
    return text.slice(0, this.opts.maxInputChars);
  }

  private splitBatches(texts: string[]): { index: number; text: string }[][] {
    const batches: { index: number; text: string }[][] = [];
    let current: { index: number; text: string }[] = [];
    let currentBytes = 0;
    texts.forEach((text, index) => {
      const bytes = Buffer.byteLength(text, "utf8");
      const wouldOverflow = current.length > 0
        && (current.length >= this.opts.maxBatchItems || currentBytes + bytes > this.opts.maxBatchBytes);
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

  private async requestWithRetry(inputs: string[]): Promise<number[][]> {
    const url = this.endpoint;
    let attempt = 0;
    for (;;) {
      if (Date.now() < this.circuitOpenUntil) {
        const remaining = Math.ceil((this.circuitOpenUntil - Date.now()) / 1000);
        throw new RemoteLLMError("circuit", url,
          `Disjoncteur ouvert vers ${url} après ${this.consecutiveFailures} échecs consécutifs ; nouvel essai dans ${remaining} s`);
      }
      try {
        const vectors = await this.requestOnce(url, inputs);
        this.consecutiveFailures = 0;
        return vectors;
      } catch (error) {
        const remoteError = error instanceof RemoteLLMError ? error : new RemoteLLMError("retryable", url, String(error));
        const retryable = remoteError.kind === "retryable";
        if (retryable) this.noteFailure();
        if (!retryable || attempt >= this.opts.maxRetries) {
          throw remoteError;
        }
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

  private retryAfterMs: number | null = null;

  private backoffMs(attempt: number, error: RemoteLLMError): number {
    const hinted = this.retryAfterMs;
    this.retryAfterMs = null;
    if (error.status === 429 && hinted !== null) return Math.min(hinted, 10_000);
    const base = 500 * 2 ** (attempt - 1);
    const jitter = Math.floor(Math.random() * 250);
    return Math.min(base + jitter, 10_000);
  }

  private async requestOnce(url: string, inputs: string[]): Promise<number[][]> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.opts.timeoutMs);
    let response: Response;
    try {
      response = await this.fetchImpl(url, {
        method: "POST",
        headers: this.headers(),
        body: JSON.stringify(this.body(inputs)),
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
          `HTTP ${status} sur ${url} pour ${this.ref.model} — clé ${this.apiKey ? "refusée" : "absente (QMD_EMBED_API_KEY / QMD_API_KEY)"}${detail ? ` : ${detail}` : ""}`, status);
      }
      if (status === 429 || status === 408 || status >= 500) {
        const retryAfter = response.headers.get("retry-after");
        if (retryAfter && /^\d+$/.test(retryAfter.trim())) this.retryAfterMs = Number(retryAfter.trim()) * 1000;
        throw new RemoteLLMError("retryable", url, `HTTP ${status}${detail ? ` : ${detail}` : ""}`, status);
      }
      throw new RemoteLLMError("http", url, `HTTP ${status} sur ${url} pour ${this.ref.model}${detail ? ` : ${detail}` : ""}`, status);
    }

    const contentType = response.headers.get("content-type") ?? "";
    const raw = await safeReadText(response, 4_000_000);
    if (!/json/i.test(contentType) && !raw.trimStart().startsWith("{")) {
      // Le cas OpenWebUI : une route inconnue rend le SPA en 200 text/html.
      throw new RemoteLLMError("contract", url,
        `Réponse non JSON (${contentType || "sans content-type"}) sur ${url} : ce n'est pas une route d'embeddings`);
    }
    let payload: unknown;
    try {
      payload = JSON.parse(raw);
    } catch {
      throw new RemoteLLMError("contract", url, `JSON illisible sur ${url}`);
    }
    const vectors = this.parseVectors(url, payload, inputs.length);
    this.checkDimensions(url, vectors);
    return vectors;
  }

  private headers(): Record<string, string> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      "Accept": "application/json",
    };
    if (this.apiKey) headers["Authorization"] = `Bearer ${this.apiKey}`;
    return headers;
  }

  private body(inputs: string[]): Record<string, unknown> {
    if (this.ref.scheme === "ollama") {
      return { model: this.ref.model, input: inputs, truncate: true };
    }
    return { model: this.ref.model, input: inputs, encoding_format: "float" };
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

async function safeReadText(response: Response, limit = 600): Promise<string> {
  try {
    const text = await response.text();
    return text.length > limit ? `${text.slice(0, limit)}…` : text;
  } catch {
    return "";
  }
}

function describeError(obj: Record<string, unknown>): string {
  const err = obj.error;
  if (!err) return "";
  if (typeof err === "string") return ` : ${err}`;
  const message = (err as Record<string, unknown>).message;
  return typeof message === "string" ? ` : ${message}` : "";
}

// =============================================================================
// RemoteLLM — le backend vu par store.ts
// =============================================================================

export type RemoteLLMConfig = {
  /** URI complète, ex. openai:https://…/v1#qwen3-embedding-8k */
  embedModel: string;
  apiKey?: string;
  client?: RemoteClientOptions;
};

/**
 * Backend d'embedding distant. Lot 1 : embeddings seulement. Les autres
 * opérations lèvent `unsupported` ; HybridLLM (llm.ts) les route vers le
 * backend local en attendant le lot 2.
 */
export class RemoteLLM implements QmdLLM {
  readonly isRemote = true;
  readonly client: RemoteEmbeddingClient;
  private readonly uri: string;

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
  }

  get embedModelName(): string {
    return this.uri;
  }

  get generateModelName(): string {
    return "";
  }

  get rerankModelName(): string {
    return "";
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
    throw new RemoteLLMError("unsupported", this.client.endpoint, "generate n'est pas servi par le backend distant (lot 2)");
  }

  async modelExists(model: string): Promise<ModelInfo> {
    return { name: model, exists: true, path: this.client.endpoint };
  }

  /** Sans modèle de génération : la requête telle quelle, en lexical et en vectoriel. */
  async expandQuery(query: string, options?: { context?: string; includeLexical?: boolean }): Promise<Queryable[]> {
    const queries: Queryable[] = [];
    if (options?.includeLexical !== false) queries.push({ type: "lex", text: query });
    queries.push({ type: "vec", text: query });
    return queries;
  }

  async rerank(_query: string, _documents: RerankDocument[], _options?: RerankOptions): Promise<RerankResult> {
    throw new RemoteLLMError("unsupported", this.client.endpoint, "rerank n'est pas servi par le backend distant (lot 2)");
  }

  async dispose(): Promise<void> {
    // Rien à libérer : pas de contexte natif.
  }
}
