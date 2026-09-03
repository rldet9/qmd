/**
 * Recette du backend d'embedding distant — fork MIXTRIO,
 * SPEC-QMD-FORK-REMOTE-2026-001, lot 1 (critères 2, 3, 4).
 *
 * Tout tourne contre un faux serveur HTTP en processus : aucun réseau, aucun
 * modèle GGUF. Le faux serveur compte les requêtes en vol pour prouver la
 * borne de concurrence (D-6), et `setNodeLlamaCppModuleForTest` fait échouer
 * tout chargement de node-llama-cpp pour prouver qu'un embedding distant ne
 * touche jamais le runtime local (critère 4).
 */

import { describe, test, expect, beforeEach, afterEach } from "vitest";
import { createServer, type Server, type IncomingMessage, type ServerResponse } from "node:http";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  RemoteEmbeddingClient,
  RemoteLLM,
  RemoteLLMError,
  parseRemoteModelUri,
  isRemoteModelUri,
  looksLikeRemoteModelUri,
  resolveRemoteApiKey,
  REMOTE_DEFAULTS,
} from "../src/remote-llm.js";
import { createLLM, HybridLLM, LlamaCpp, setNodeLlamaCppModuleForTest, setDefaultLlamaCpp } from "../src/llm.js";
import { createStore } from "../src/index.js";
import { chunkDocumentByTokens, getEmbeddingFingerprint } from "../src/store.js";

// =============================================================================
// Faux serveur
// =============================================================================

type Recorded = {
  path: string;
  headers: Record<string, string | string[] | undefined>;
  body: any;
};

type FakeOptions = {
  dims?: number;
  /** Séquence de réponses forcées, consommée requête par requête (status, body?). */
  script?: Array<{ status: number; body?: string; headers?: Record<string, string> }>;
  delayMs?: number;
  /** Dimension différente à partir de la N-ième requête (critère 3). */
  dimsAfter?: { request: number; dims: number };
};

type Fake = {
  url: string;
  requests: Recorded[];
  maxInFlight: number;
  close: () => Promise<void>;
};

async function startFake(options: FakeOptions = {}): Promise<Fake> {
  const dims = options.dims ?? 8;
  const script = [...(options.script ?? [])];
  const requests: Recorded[] = [];
  let inFlight = 0;
  let maxInFlight = 0;

  const server: Server = createServer((req: IncomingMessage, res: ServerResponse) => {
    inFlight++;
    maxInFlight = Math.max(maxInFlight, inFlight);
    let raw = "";
    req.on("data", (chunk) => { raw += chunk; });
    req.on("end", async () => {
      const body = raw ? JSON.parse(raw) : null;
      requests.push({ path: req.url ?? "", headers: req.headers, body });
      if (options.delayMs) await new Promise((resolve) => setTimeout(resolve, options.delayMs));

      const forced = script.shift();
      if (forced) {
        res.writeHead(forced.status, { "content-type": "application/json", ...(forced.headers ?? {}) });
        res.end(forced.body ?? JSON.stringify({ error: { message: `forced ${forced.status}` } }));
        inFlight--;
        return;
      }

      const inputs: string[] = Array.isArray(body?.input) ? body.input : [body?.input];
      const useDims = options.dimsAfter && requests.length >= options.dimsAfter.request ? options.dimsAfter.dims : dims;
      const vector = (i: number) => Array.from({ length: useDims }, (_, k) => (i + 1) * 0.01 + k * 0.001);

      if ((req.url ?? "").endsWith("/api/embed")) {
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify({ model: body.model, embeddings: inputs.map((_, i) => vector(i)), prompt_eval_count: inputs.length }));
      } else {
        // Réponse OpenAI, volontairement dans le désordre pour vérifier le tri par index.
        const data = inputs.map((_, i) => ({ object: "embedding", index: i, embedding: vector(i) })).reverse();
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify({ object: "list", model: body.model, data, usage: { prompt_tokens: 1, total_tokens: 1 } }));
      }
      inFlight--;
    });
  });

  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  const port = typeof address === "object" && address ? address.port : 0;
  return {
    url: `http://127.0.0.1:${port}`,
    requests,
    get maxInFlight() { return maxInFlight; },
    close: () => new Promise<void>((resolve) => server.close(() => resolve())),
  };
}

const noSleep = async () => {};
const quiet = () => {};

// =============================================================================
// URI
// =============================================================================

describe("parseRemoteModelUri", () => {
  test("openai: avec base /v1 et modèle", () => {
    const ref = parseRemoteModelUri("openai:https://gateway.example/v1#qwen3-embedding-8k");
    expect(ref).toEqual({
      scheme: "openai",
      baseUrl: "https://gateway.example/v1",
      model: "qwen3-embedding-8k",
      uri: "openai:https://gateway.example/v1#qwen3-embedding-8k",
    });
  });

  test("ollama: avec tag de modèle et barre finale retirée", () => {
    const ref = parseRemoteModelUri("ollama:http://ollama.internal:11434/#qwen3-embedding-8k:latest");
    expect(ref?.scheme).toBe("ollama");
    expect(ref?.baseUrl).toBe("http://ollama.internal:11434");
    expect(ref?.model).toBe("qwen3-embedding-8k:latest");
  });

  test.each([
    "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf",
    "/models/local.gguf",
    "",
  ])("%s n'est pas distant", (uri) => {
    expect(isRemoteModelUri(uri)).toBe(false);
    expect(looksLikeRemoteModelUri(uri)).toBe(false);
  });

  test.each([
    "openai:qwen3-embedding-8k",           // pas d'URL
    "openai:https://host/v1",              // pas de modèle
    "ollama:ftp://host#model",             // schéma d'URL non http
  ])("%s ressemble à du distant mais est invalide", (uri) => {
    expect(looksLikeRemoteModelUri(uri)).toBe(true);
    expect(parseRemoteModelUri(uri)).toBeNull();
  });

  test("createLLM refuse une URI distante mal formée au lieu de retomber sur le GGUF", () => {
    expect(() => createLLM({ embedModel: "openai:qwen3-embedding-8k" })).toThrow(/URI de modèle distant invalide/);
  });

  test("createLLM rend LlamaCpp pour une URI hf: et HybridLLM pour une URI distante", () => {
    expect(createLLM({ embedModel: "hf:ggml-org/x/y.gguf" })).toBeInstanceOf(LlamaCpp);
    const hybrid = createLLM({ embedModel: "openai:http://127.0.0.1:9/v1#m" });
    expect(hybrid).toBeInstanceOf(HybridLLM);
    expect(hybrid.isRemote).toBe(true);
    expect(hybrid.embedModelName).toBe("openai:http://127.0.0.1:9/v1#m");
    expect((hybrid as HybridLLM).hasLocal).toBe(false);
  });
});

describe("resolveRemoteApiKey (D-4)", () => {
  test("clé de rôle, puis clé partagée, jamais rien d'autre", () => {
    expect(resolveRemoteApiKey("embed", { QMD_EMBED_API_KEY: "role", QMD_API_KEY: "shared" })).toBe("role");
    expect(resolveRemoteApiKey("embed", { QMD_API_KEY: "shared" })).toBe("shared");
    expect(resolveRemoteApiKey("embed", { QMD_EMBED_API_KEY: "  ", QMD_API_KEY: "" })).toBeUndefined();
    expect(resolveRemoteApiKey("rerank", { QMD_EMBED_API_KEY: "role" })).toBeUndefined();
  });
});

// =============================================================================
// Client
// =============================================================================

describe("RemoteEmbeddingClient — OpenAI-compatible", () => {
  let fake: Fake;
  afterEach(async () => { await fake?.close(); });

  test("lots ≤ 32 textes, Authorization Bearer sur chaque requête, vecteurs dans l'ordre (critère 2)", async () => {
    fake = await startFake({ dims: 4 });
    const client = new RemoteEmbeddingClient(parseRemoteModelUri(`openai:${fake.url}/v1#qwen3-embedding-8k`)!, {
      apiKey: "sk-test", sleep: noSleep, warn: quiet,
    });
    const texts = Array.from({ length: 70 }, (_, i) => `texte ${i}`);
    const vectors = await client.embedBatch(texts);

    expect(vectors).toHaveLength(70);
    expect(vectors.every((v) => v.length === 4)).toBe(true);
    // L'ordre est celui des textes, malgré une réponse serveur inversée.
    expect(vectors[0]![0]).toBeCloseTo(0.01, 6);
    expect(vectors[1]![0]).toBeCloseTo(0.02, 6);
    expect(fake.requests).toHaveLength(3);
    expect(fake.requests.map((r) => r.body.input.length)).toEqual([32, 32, 6]);
    for (const r of fake.requests) {
      expect(r.path).toBe("/v1/embeddings");
      expect(r.headers.authorization).toBe("Bearer sk-test");
      expect(r.body.model).toBe("qwen3-embedding-8k");
    }
    expect(client.knownDimensions).toBe(4);
  });

  test("≤ 64 Ko par requête", async () => {
    fake = await startFake();
    const client = new RemoteEmbeddingClient(parseRemoteModelUri(`openai:${fake.url}/v1#m`)!, { sleep: noSleep, warn: quiet });
    const texts = Array.from({ length: 7 }, () => "x".repeat(20 * 1024));
    await client.embedBatch(texts);
    expect(fake.requests.length).toBeGreaterThanOrEqual(3);
    for (const r of fake.requests) {
      const bytes = r.body.input.reduce((sum: number, t: string) => sum + Buffer.byteLength(t), 0);
      expect(bytes).toBeLessThanOrEqual(REMOTE_DEFAULTS.maxBatchBytes);
    }
  });

  test("jamais plus de 4 requêtes en vol (D-6), et bien en parallèle", async () => {
    fake = await startFake({ delayMs: 40 });
    const client = new RemoteEmbeddingClient(parseRemoteModelUri(`openai:${fake.url}/v1#m`)!, { sleep: noSleep, warn: quiet });
    const texts = Array.from({ length: 32 * 10 }, (_, i) => `t${i}`);
    await client.embedBatch(texts);
    expect(fake.requests).toHaveLength(10);
    expect(fake.maxInFlight).toBeLessThanOrEqual(4);
    expect(fake.maxInFlight).toBeGreaterThanOrEqual(2);
  });

  test("429 puis 200 : une reprise, puis succès", async () => {
    fake = await startFake({ script: [{ status: 429, headers: { "retry-after": "1" } }] });
    const client = new RemoteEmbeddingClient(parseRemoteModelUri(`openai:${fake.url}/v1#m`)!, { sleep: noSleep, warn: quiet });
    const vectors = await client.embedBatch(["a"]);
    expect(vectors).toHaveLength(1);
    expect(fake.requests).toHaveLength(2);
  });

  test("5xx persistant : échec après maxRetries reprises, kind retryable", async () => {
    fake = await startFake({ script: Array.from({ length: 10 }, () => ({ status: 503 })) });
    const client = new RemoteEmbeddingClient(parseRemoteModelUri(`openai:${fake.url}/v1#m`)!, { sleep: noSleep, warn: quiet, maxRetries: 2 });
    await expect(client.embedBatch(["a"])).rejects.toMatchObject({ name: "RemoteLLMError", kind: "retryable", status: 503 });
    expect(fake.requests).toHaveLength(3);
  });

  test("401 : erreur d'authentification nommée, une seule requête, pas de reprise (critère 4)", async () => {
    fake = await startFake({ script: [{ status: 401, body: JSON.stringify({ error: { message: "Unable to decode token" } }) }] });
    const client = new RemoteEmbeddingClient(parseRemoteModelUri(`openai:${fake.url}/v1#qwen`)!, { apiKey: "bad", sleep: noSleep, warn: quiet });
    const error = await client.embedBatch(["a"]).catch((e) => e);
    expect(error).toBeInstanceOf(RemoteLLMError);
    expect(error.kind).toBe("auth");
    expect(error.message).toMatch(/HTTP 401/);
    expect(error.message).toMatch(/qwen/);
    expect(error.message).toMatch(/refusée/);
    expect(fake.requests).toHaveLength(1);
  });

  test("401 sans clé : le message dit quelle variable poser", async () => {
    fake = await startFake({ script: [{ status: 401 }] });
    const client = new RemoteEmbeddingClient(parseRemoteModelUri(`openai:${fake.url}/v1#m`)!, { sleep: noSleep, warn: quiet });
    await expect(client.embedBatch(["a"])).rejects.toThrow(/QMD_EMBED_API_KEY/);
    expect(fake.requests[0]!.headers.authorization).toBeUndefined();
  });

  test("dimension qui change d'un appel à l'autre : erreur nommant les deux dimensions (critère 3)", async () => {
    fake = await startFake({ dims: 1024, dimsAfter: { request: 2, dims: 4096 } });
    const client = new RemoteEmbeddingClient(parseRemoteModelUri(`openai:${fake.url}/v1#m`)!, { sleep: noSleep, warn: quiet });
    await client.embedBatch(["a"]);
    const error = await client.embedBatch(["b"]).catch((e) => e);
    expect(error.kind).toBe("dimension");
    expect(error.message).toMatch(/4096/);
    expect(error.message).toMatch(/1024/);
  });

  test("réponse HTML en 200 : erreur de contrat, pas un vecteur", async () => {
    fake = await startFake({ script: [{ status: 200, headers: { "content-type": "text/html" }, body: "<!doctype html><html>SPA</html>" }] });
    const client = new RemoteEmbeddingClient(parseRemoteModelUri(`openai:${fake.url}/v1#m`)!, { sleep: noSleep, warn: quiet });
    const error = await client.embedBatch(["a"]).catch((e) => e);
    expect(error.kind).toBe("contract");
    expect(error.message).toMatch(/non JSON/);
  });

  test("disjoncteur : après 5 échecs consécutifs les appels échouent sans requête", async () => {
    fake = await startFake({ script: Array.from({ length: 20 }, () => ({ status: 500 })) });
    const client = new RemoteEmbeddingClient(parseRemoteModelUri(`openai:${fake.url}/v1#m`)!, { sleep: noSleep, warn: quiet, maxRetries: 4 });
    await expect(client.embedBatch(["a"])).rejects.toMatchObject({ kind: "retryable" });
    const before = fake.requests.length;
    await expect(client.embedBatch(["b"])).rejects.toMatchObject({ kind: "circuit" });
    expect(fake.requests.length).toBe(before);
  });

  test("texte au-delà du plafond de caractères : tronqué, avec un avertissement (D-8)", async () => {
    fake = await startFake();
    const warnings: string[] = [];
    const client = new RemoteEmbeddingClient(parseRemoteModelUri(`openai:${fake.url}/v1#m`)!, { sleep: noSleep, warn: (m) => warnings.push(m), maxInputChars: 100 });
    await client.embedBatch(["y".repeat(500)]);
    expect(fake.requests[0]!.body.input[0]).toHaveLength(100);
    expect(warnings.some((w) => /tronqué/.test(w))).toBe(true);
  });

  test("probe : une requête, la dimension et une latence", async () => {
    fake = await startFake({ dims: 16 });
    const client = new RemoteEmbeddingClient(parseRemoteModelUri(`openai:${fake.url}/v1#m`)!, { sleep: noSleep, warn: quiet });
    const probe = await client.probe();
    expect(probe.dimensions).toBe(16);
    expect(probe.latencyMs).toBeGreaterThanOrEqual(0);
    expect(client.describe()).toMatchObject({ scheme: "openai", model: "m", hasApiKey: false, dimensions: 16, endpoint: `${fake.url}/v1/embeddings` });
  });
});

describe("RemoteEmbeddingClient — Ollama natif", () => {
  let fake: Fake;
  afterEach(async () => { await fake?.close(); });

  test("POST /api/embed avec truncate, sans Authorization même si une clé est fournie (D-4)", async () => {
    fake = await startFake({ dims: 1024 });
    const client = new RemoteEmbeddingClient(parseRemoteModelUri(`ollama:${fake.url}#qwen3-embedding-8k:latest`)!, { apiKey: "ignored", sleep: noSleep, warn: quiet });
    const vectors = await client.embedBatch(["a", "b"]);
    expect(vectors).toHaveLength(2);
    expect(vectors[0]).toHaveLength(1024);
    expect(fake.requests[0]!.path).toBe("/api/embed");
    expect(fake.requests[0]!.body).toMatchObject({ model: "qwen3-embedding-8k:latest", input: ["a", "b"], truncate: true });
    expect(fake.requests[0]!.headers.authorization).toBeUndefined();
    expect(client.describe().hasApiKey).toBe(false);
  });
});

// =============================================================================
// RemoteLLM / HybridLLM vus par le store
// =============================================================================

describe("RemoteLLM", () => {
  let fake: Fake;
  afterEach(async () => { await fake?.close(); });

  test("embed rend {embedding, model: URI} et embedModelName est l'URI complète", async () => {
    fake = await startFake({ dims: 3 });
    const uri = `openai:${fake.url}/v1#m`;
    const llm = new RemoteLLM({ embedModel: uri, client: { sleep: noSleep, warn: quiet } });
    const result = await llm.embed("bonjour");
    expect(result?.model).toBe(uri);
    expect(result?.embedding).toHaveLength(3);
    expect(llm.embedModelName).toBe(uri);
    expect(llm.isRemote).toBe(true);
    expect(llm.tokenize).toBeUndefined();
    await expect(llm.countTokens("abcdef")).resolves.toBe(2);
  });

  test("rerank et generate lèvent unsupported (lot 2), expandQuery rend lex + vec", async () => {
    fake = await startFake();
    const llm = new RemoteLLM({ embedModel: `openai:${fake.url}/v1#m`, client: { sleep: noSleep, warn: quiet } });
    await expect(llm.rerank("q", [{ file: "a", text: "t" }])).rejects.toMatchObject({ kind: "unsupported" });
    await expect(llm.generate("p")).rejects.toMatchObject({ kind: "unsupported" });
    await expect(llm.expandQuery("clé virtuelle")).resolves.toEqual([
      { type: "lex", text: "clé virtuelle" },
      { type: "vec", text: "clé virtuelle" },
    ]);
  });

  test("le fingerprint d'une URI qwen3 distante utilise le format Qwen3 et diffère du défaut", () => {
    const uri = "openai:https://gw/v1#qwen3-embedding-8k";
    const other = "openai:https://gw/v1#qwen3-embedding-8b";
    expect(getEmbeddingFingerprint(uri)).not.toBe(getEmbeddingFingerprint());
    expect(getEmbeddingFingerprint(uri)).not.toBe(getEmbeddingFingerprint(other));
  });
});

describe("chunkDocumentByTokens sans tokenizer (D-8)", () => {
  test("découpe en caractères, ≤ 2 700 caractères par chunk, sans toucher au singleton local", async () => {
    const llm = new RemoteLLM({ embedModel: "openai:http://127.0.0.1:9/v1#m", client: { sleep: noSleep, warn: quiet } });
    const text = Array.from({ length: 400 }, (_, i) => `Ligne ${i} d'un document de recette assez long pour être découpé.`).join("\n");
    const chunks = await chunkDocumentByTokens(text, undefined, undefined, undefined, "doc.md", "regex", undefined, llm);
    expect(chunks.length).toBeGreaterThan(1);
    for (const chunk of chunks) {
      expect(chunk.text.length).toBeLessThanOrEqual(900 * 3);
      expect(chunk.tokens).toBe(Math.ceil(chunk.text.length / 3));
    }
    expect(chunks.map((c) => c.text).join("").length).toBeGreaterThan(0);
  });
});

// =============================================================================
// De bout en bout : createStore + embed sur un faux serveur, node-llama-cpp interdit
// =============================================================================

describe("createStore avec models.embed distant", () => {
  let fake: Fake;
  let dir: string;
  const forbidden = new Error("node-llama-cpp ne doit jamais être chargé pour un embedding distant");

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "qmd-remote-"));
    writeFileSync(join(dir, "gateway.md"), "# Gateway LiteLLM\n\nLes clés virtuelles sont provisionnées par setup-llm-gateway.ps1.\n");
    writeFileSync(join(dir, "ollama.md"), "# Ollama GPU\n\nqwen3-embedding-8k rend des vecteurs de 1024 dimensions.\n");
    setNodeLlamaCppModuleForTest({
      getLlama: async () => { throw forbidden; },
      resolveModelFile: async () => { throw forbidden; },
      LlamaChatSession: class { constructor() { throw forbidden; } prompt = async () => ""; } as any,
      LlamaLogLevel: { error: 0 },
    });
  });

  afterEach(async () => {
    setNodeLlamaCppModuleForTest(null);
    setDefaultLlamaCpp(null);
    await fake?.close();
    rmSync(dir, { recursive: true, force: true });
  });

  test("embed écrit l'URI dans content_vectors.model, crée vectors_vec à la bonne dimension, et ne charge aucun GGUF", async () => {
    fake = await startFake({ dims: 8 });
    const uri = `openai:${fake.url}/v1#qwen3-embedding-8k`;
    process.env.QMD_API_KEY = "sk-recette";
    try {
      const store = await createStore({
        dbPath: join(dir, "index.sqlite"),
        config: { collections: { docs: { path: dir, pattern: "**/*.md" } }, models: { embed: uri } },
      });
      await store.update();
      const result = await store.embed();
      expect(result.errors).toBe(0);
      expect(result.chunksEmbedded).toBeGreaterThanOrEqual(2);

      const rows = store.internal.db.prepare(`SELECT DISTINCT model, embed_fingerprint FROM content_vectors`).all() as { model: string; embed_fingerprint: string }[];
      expect(rows).toHaveLength(1);
      expect(rows[0]!.model).toBe(uri);
      expect(rows[0]!.embed_fingerprint).toBe(getEmbeddingFingerprint(uri));

      const vec = store.internal.db.prepare(`SELECT sql FROM sqlite_master WHERE name = 'vectors_vec'`).get() as { sql: string };
      expect(vec.sql).toMatch(/float\[8\]/);

      // Les documents sont partis au format Qwen3 : texte brut précédé du titre, jamais "title: … | text: …".
      const sent: string[] = fake.requests.flatMap((r) => r.body.input);
      expect(sent.some((t) => t.startsWith("Gateway LiteLLM\n"))).toBe(true);
      expect(sent.every((t) => !t.startsWith("title:"))).toBe(true);
      for (const r of fake.requests) expect(r.headers.authorization).toBe("Bearer sk-recette");

      // Une requête vectorielle passe aussi par le distant, au format d'instruction Qwen3.
      const hits = await store.searchVector("clés virtuelles", { limit: 2 });
      expect(hits.length).toBeGreaterThan(0);
      const last = fake.requests.at(-1)!.body.input[0] as string;
      expect(last.startsWith("Instruct: ")).toBe(true);

      const status = await store.getStatus();
      expect(status.needsEmbedding).toBe(0);
      await store.close();
    } finally {
      delete process.env.QMD_API_KEY;
    }
  });

  test("401 au premier chunk : embed échoue explicitement et n'écrit aucun vecteur (critère 4)", async () => {
    fake = await startFake({ script: [{ status: 401 }] });
    const uri = `openai:${fake.url}/v1#qwen3-embedding-8k`;
    const store = await createStore({
      dbPath: join(dir, "index.sqlite"),
      config: { collections: { docs: { path: dir, pattern: "**/*.md" } }, models: { embed: uri } },
    });
    await store.update();
    await expect(store.embed()).rejects.toThrow(/HTTP 401/);
    const count = store.internal.db.prepare(`SELECT COUNT(*) AS n FROM content_vectors`).get() as { n: number };
    expect(count.n).toBe(0);
    const vec = store.internal.db.prepare(`SELECT 1 FROM sqlite_master WHERE name = 'vectors_vec'`).get();
    expect(vec).toBeUndefined();
    await store.close();
  });
});
