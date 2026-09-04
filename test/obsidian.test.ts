/**
 * Recette des coffres Obsidian — fork MIXTRIO,
 * SPEC-QMD-FORK-REMOTE-2026-001, lot 3-bis (D-12 et D-13).
 *
 * Deux niveaux : les fonctions pures de `src/obsidian.ts`, puis un coffre de
 * recette réellement indexé, pour vérifier le titre issu du frontmatter, le
 * graphe dans les deux sens, et les exclusions `.obsidian` / `.trash`.
 */

import { describe, test, expect, beforeAll, afterAll } from "vitest";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  parseFrontmatter,
  frontmatterTitle,
  extractWikilinks,
  normalizeWikilinks,
  shortName,
  targetCandidates,
  OBSIDIAN_PIPELINE_VERSION,
} from "../src/obsidian.js";
import { createStore } from "../src/index.js";
import { getLinksIn, getLinksOut, getLinkStats, extractTitle, getEmbeddingFingerprint } from "../src/store.js";

// =============================================================================
// Frontmatter (D-12)
// =============================================================================

describe("parseFrontmatter", () => {
  test("titre, alias en liste multi-lignes, tags en ligne", () => {
    const { frontmatter, body } = parseFrontmatter(
      "---\ntitle: Gateway LiteLLM\naliases:\n  - gateway\n  - passerelle LLM\ntags: [infra, llm]\n---\n# Autre titre\n\nCorps.\n",
    );
    expect(frontmatter?.title).toBe("Gateway LiteLLM");
    expect(frontmatter?.aliases).toEqual(["gateway", "passerelle LLM"]);
    expect(frontmatter?.tags).toEqual(["infra", "llm"]);
    expect(body.startsWith("# Autre titre")).toBe(true);
  });

  test("guillemets retirés, clés scalaires conservées, structures ignorées", () => {
    const { frontmatter } = parseFrontmatter(
      '---\ntitle: "Clés virtuelles"\ndescription: \'Six clés\'\nmetadata: {a: 1}\nbloc: |\n  ligne\n---\ntexte\n',
    );
    expect(frontmatter?.title).toBe("Clés virtuelles");
    expect(frontmatter?.extra.description).toBe("Six clés");
    expect(frontmatter?.extra.metadata).toBeUndefined();
    expect(frontmatter?.extra.bloc).toBeUndefined();
  });

  test("sans frontmatter, le corps est rendu intact", () => {
    const content = "# Titre\n\nUn texte avec --- au milieu\n\n---\n\nsuite\n";
    const { frontmatter, body } = parseFrontmatter(content);
    expect(frontmatter).toBeNull();
    expect(body).toBe(content);
  });

  test("un délimiteur non fermé n'est pas un frontmatter", () => {
    const content = "---\ntitle: jamais fermé\n\n# Titre\n";
    expect(parseFrontmatter(content).frontmatter).toBeNull();
  });

  test("frontmatterTitle rend null quand il n'y a pas de titre déclaré", () => {
    expect(frontmatterTitle("---\ntags: [a]\n---\n# Vrai titre\n")).toBeNull();
    expect(frontmatterTitle("# Vrai titre\n")).toBeNull();
    expect(frontmatterTitle("---\ntitle: Déclaré\n---\n# Vrai titre\n")).toBe("Déclaré");
  });
});

describe("extractTitle — le frontmatter fait autorité (D-12)", () => {
  test("title: prime sur le premier titre markdown", () => {
    expect(extractTitle("---\ntitle: Déclaré\n---\n# Markdown\n", "note.md")).toBe("Déclaré");
  });

  test("sans frontmatter, la règle amont est inchangée", () => {
    expect(extractTitle("# Markdown\n\ntexte", "note.md")).toBe("Markdown");
    expect(extractTitle("pas de titre", "dossier/note.md")).toBe("note");
  });
});

describe("empreinte d'embedding", () => {
  test("la version du pipeline entre dans l'empreinte", () => {
    // Le titre est vectorisé avec le texte : lire le frontmatter change les
    // vecteurs, l'empreinte doit donc changer aussi.
    expect(OBSIDIAN_PIPELINE_VERSION).toBeTruthy();
    expect(getEmbeddingFingerprint("modele-a")).not.toBe(getEmbeddingFingerprint("modele-b"));
    expect(getEmbeddingFingerprint()).toMatch(/^[0-9a-f]{6}$/);
  });
});

// =============================================================================
// Wikilinks (D-13)
// =============================================================================

describe("extractWikilinks", () => {
  test("alias, ancre, embed et extension .md", () => {
    const links = extractWikilinks(
      "Voir [[carte/env/M3_REDIS_URL|M3_REDIS_URL]] et [[note simple]].\n" +
      "Section : [[guide#Installation]]. Embed : ![[schema.md]].",
    );
    expect(links).toEqual([
      { target: "carte/env/M3_REDIS_URL", anchor: null, alias: "M3_REDIS_URL", embed: false },
      { target: "note simple", anchor: null, alias: null, embed: false },
      { target: "guide", anchor: "Installation", alias: null, embed: false },
      { target: "schema", anchor: null, alias: null, embed: true },
    ]);
  });

  test("un texte sans wikilink rend une liste vide", () => {
    expect(extractWikilinks("# Titre\n\n[lien markdown](http://exemple)\n")).toEqual([]);
  });

  test("les crochets isolés ne sont pas des wikilinks", () => {
    expect(extractWikilinks("un tableau [0] et [[]] vide")).toEqual([]);
  });
});

describe("normalizeWikilinks — affichage seulement", () => {
  test("porte le libellé ET le chemin cible", () => {
    expect(normalizeWikilinks("voir [[carte/env/X|X]] ici")).toBe("voir X (→ carte/env/X) ici");
    expect(normalizeWikilinks("voir [[carte/env/X]] ici")).toBe("voir X (→ carte/env/X) ici");
    expect(normalizeWikilinks("voir [[WikiWord]] ici")).toBe("voir WikiWord ici");
    expect(normalizeWikilinks("voir [[guide#Section]]")).toBe("voir guide (section Section)");
  });
});

describe("shortName et targetCandidates", () => {
  test("le nom court est le dernier segment", () => {
    expect(shortName("carte/env/X")).toBe("X");
    expect(shortName("X")).toBe("X");
  });

  test("les candidats couvrent le chemin exact puis le nom court", () => {
    expect(targetCandidates("carte/env/X")).toEqual({
      exact: ["carte/env/X.md", "carte/env/X"],
      suffix: "/X.md",
    });
    expect(targetCandidates("carte/env/X.md").exact[0]).toBe("carte/env/X.md");
  });
});

// =============================================================================
// De bout en bout : un coffre de recette réellement indexé
// =============================================================================

describe("graphe d'un coffre Obsidian indexé", () => {
  let dir: string;
  let store: Awaited<ReturnType<typeof createStore>>;

  beforeAll(async () => {
    dir = mkdtempSync(join(tmpdir(), "qmd-vault-"));
    mkdirSync(join(dir, "carte", "env"), { recursive: true });
    mkdirSync(join(dir, ".obsidian"), { recursive: true });
    mkdirSync(join(dir, ".trash"), { recursive: true });

    writeFileSync(join(dir, "carte", "env", "M3_REDIS_URL.md"),
      "---\ntitle: M3_REDIS_URL\naliases:\n  - url redis\n---\n\nVariable du compose.\n\n" +
      "Services : [[carte/service/n8n|n8n]] et [[carte/service/owui|owui]].\n");

    mkdirSync(join(dir, "carte", "service"), { recursive: true });
    writeFileSync(join(dir, "carte", "service", "n8n.md"),
      "# n8n\n\nLit [[carte/env/M3_REDIS_URL|M3_REDIS_URL]] et [[carte/env/ABSENTE|ABSENTE]].\n");
    writeFileSync(join(dir, "carte", "service", "owui.md"),
      "# owui\n\nLit [[carte/env/M3_REDIS_URL]].\n");

    writeFileSync(join(dir, "note-simple.md"), "# Note simple\n\nAucun lien ici.\n");
    // Ces deux-là ne doivent JAMAIS être indexés.
    writeFileSync(join(dir, ".obsidian", "workspace.md"), "# Config du coffre\n");
    writeFileSync(join(dir, ".trash", "supprime.md"), "# Note supprimée\n\n[[carte/env/M3_REDIS_URL]]\n");

    store = await createStore({
      dbPath: join(dir, "index.sqlite"),
      config: { collections: { vault: { path: dir, pattern: "**/*.md" } } },
    });
    await store.update();
  });

  afterAll(async () => {
    await store?.close();
    rmSync(dir, { recursive: true, force: true });
  });

  test("`.obsidian` et `.trash` ne sont pas indexés (D-12)", () => {
    const paths = store.internal.db
      .prepare(`SELECT path FROM documents WHERE active = 1 ORDER BY path`)
      .all() as { path: string }[];
    const list = paths.map((p) => p.path);
    expect(list).toContain("carte/env/M3_REDIS_URL.md");
    expect(list).toContain("note-simple.md");
    expect(list.some((p) => p.includes(".obsidian"))).toBe(false);
    expect(list.some((p) => p.includes(".trash"))).toBe(false);
  });

  test("le titre vient du frontmatter quand il existe", () => {
    const row = store.internal.db
      .prepare(`SELECT title FROM documents WHERE path = ?`)
      .get("carte/env/M3_REDIS_URL.md") as { title: string };
    expect(row.title).toBe("M3_REDIS_URL");
  });

  test("liens SORTANTS : résolus vers les documents cibles", () => {
    const out = getLinksOut(store.internal.db, "vault", "carte/env/M3_REDIS_URL.md");
    expect(out.map((l) => l.target).sort()).toEqual(["carte/service/n8n", "carte/service/owui"]);
    expect(out.every((l) => l.path !== null)).toBe(true);
    expect(out.find((l) => l.target === "carte/service/n8n")?.alias).toBe("n8n");
  });

  test("liens ENTRANTS : les deux services qui citent la variable", () => {
    const incoming = getLinksIn(store.internal.db, "vault", "carte/env/M3_REDIS_URL.md");
    expect(incoming.map((l) => l.path).sort()).toEqual(["carte/service/n8n.md", "carte/service/owui.md"]);
    // Le lien avec alias et celui sans alias sont tous deux retrouvés.
    expect(incoming.some((l) => l.alias === "M3_REDIS_URL")).toBe(true);
    expect(incoming.some((l) => l.alias === null)).toBe(true);
  });

  test("un lien vers une note inexistante est rendu NON RÉSOLU, pas ignoré", () => {
    const out = getLinksOut(store.internal.db, "vault", "carte/service/n8n.md");
    const absente = out.find((l) => l.target === "carte/env/ABSENTE");
    expect(absente).toBeDefined();
    expect(absente!.path).toBeNull();
    expect(absente!.alias).toBe("ABSENTE");
  });

  test("un document sans lien rend une liste vide dans les deux sens", () => {
    expect(getLinksOut(store.internal.db, "vault", "note-simple.md")).toEqual([]);
    expect(getLinksIn(store.internal.db, "vault", "note-simple.md")).toEqual([]);
  });

  test("les statistiques comptent le lien non résolu", () => {
    const stats = getLinkStats(store.internal.db);
    expect(stats.total).toBe(5);
    expect(stats.documents).toBe(3);
    expect(stats.unresolved).toBe(1);
  });

  test("le chemin rendu par findDocument est utilisable tel quel pour le graphe", async () => {
    // Le CLI et le serveur MCP passent par findDocument : ce test couvre le
    // chemin d'INTÉGRATION, là où les autres appellent getLinksOut directement
    // avec le bon chemin — et n'auraient donc jamais vu le préfixe en trop.
    const { documentGraphKey } = await import("../src/store.js");
    const doc = store.internal.findDocument("carte/env/M3_REDIS_URL.md", { includeBody: false });
    expect("error" in doc).toBe(false);
    if ("error" in doc) return;
    const key = documentGraphKey(doc);
    expect(getLinksOut(store.internal.db, key.collection, key.path)).toHaveLength(2);
    expect(getLinksIn(store.internal.db, key.collection, key.path)).toHaveLength(2);
  });

  test("réindexer ne duplique pas les liens", async () => {
    const before = getLinkStats(store.internal.db).total;
    await store.update();
    expect(getLinkStats(store.internal.db).total).toBe(before);
  });
});

describe("documentGraphKey — le préfixe de collection (bug du 2026-09-04)", () => {
  test("displayPath porte la collection, la table links non", async () => {
    const { documentGraphKey } = await import("../src/store.js");
    expect(documentGraphKey({ collectionName: "carte", displayPath: "carte/env/X.md" }))
      .toEqual({ collection: "carte", path: "env/X.md" });
    // Un displayPath déjà relatif est laissé tel quel.
    expect(documentGraphKey({ collectionName: "carte", displayPath: "env/X.md" }))
      .toEqual({ collection: "carte", path: "env/X.md" });
    // Une collection homonyme d'un dossier ne doit pas être coupée deux fois.
    expect(documentGraphKey({ collectionName: "carte", displayPath: "carte/carte/X.md" }))
      .toEqual({ collection: "carte", path: "carte/X.md" });
  });
});
