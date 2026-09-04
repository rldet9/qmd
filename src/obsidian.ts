/**
 * obsidian.ts — coffres Obsidian : frontmatter, wikilinks, graphe (fork MIXTRIO).
 *
 * SPEC-QMD-FORK-REMOTE-2026-001, lot 3-bis (décisions D-12 et D-13).
 *
 * QMD 2.8.3 ignore complètement ces deux conventions : le titre d'un document
 * est son premier `#` (le `title:` du frontmatter n'est jamais lu), et un
 * `[[lien|alias]]` n'est que du texte dont la cible n'est jamais résolue.
 * Mesuré le 2026-09-03 sur nos dépôts : 1 507 wikilinks et 428 documents à
 * frontmatter dans `m3-orionai-core`, plus les 574 notes de sa carto générée,
 * dont les liens SONT l'information.
 *
 * Ce fichier est autonome : aucun import du reste de QMD, donc testable seul
 * et sans effet sur le graphe de modules.
 */

// =============================================================================
// Frontmatter (D-12)
// =============================================================================

export type Frontmatter = {
  title?: string;
  aliases: string[];
  tags: string[];
  /** Les autres clés scalaires du frontmatter. */
  extra: Record<string, string>;
};

export type ParsedDocument = {
  frontmatter: Frontmatter | null;
  /** Le corps, frontmatter retiré. */
  body: string;
};

/** Retire les guillemets encadrants et les espaces. */
function unquote(value: string): string {
  const trimmed = value.trim();
  if (trimmed.length >= 2) {
    const first = trimmed[0];
    const last = trimmed[trimmed.length - 1];
    if ((first === '"' && last === '"') || (first === "'" && last === "'")) {
      return trimmed.slice(1, -1).trim();
    }
  }
  return trimmed;
}

/** `[a, b]` ou `a, b` → ["a", "b"]. */
function parseInlineList(value: string): string[] {
  let inner = value.trim();
  if (inner.startsWith("[") && inner.endsWith("]")) inner = inner.slice(1, -1);
  return inner.split(",").map(unquote).filter(Boolean);
}

/**
 * Analyse le frontmatter YAML d'un document, sans dépendre d'un parseur YAML
 * complet : on ne lit que des scalaires et des listes, ce qui couvre `title`,
 * `aliases` et `tags`. Une structure imbriquée est ignorée plutôt que de faire
 * échouer l'indexation — un document mal formé doit rester cherchable.
 */
export function parseFrontmatter(content: string): ParsedDocument {
  // Le délimiteur doit ouvrir le fichier (un `---` au milieu est une ligne
  // horizontale markdown, pas un frontmatter).
  const normalized = content.startsWith("﻿") ? content.slice(1) : content;
  if (!/^---\r?\n/.test(normalized)) return { frontmatter: null, body: content };

  const end = normalized.search(/\r?\n---[ \t]*(\r?\n|$)/);
  if (end === -1) return { frontmatter: null, body: content };

  const raw = normalized.slice(4, end);
  const afterMatch = /\r?\n---[ \t]*(\r?\n|$)/.exec(normalized.slice(end));
  const body = normalized.slice(end + (afterMatch ? afterMatch[0].length : 0));

  const fm: Frontmatter = { aliases: [], tags: [], extra: {} };
  const lines = raw.split(/\r?\n/);
  let listKey: string | null = null;

  for (const line of lines) {
    if (!line.trim() || line.trim().startsWith("#")) continue;

    // Élément d'une liste sur plusieurs lignes : `  - valeur`
    const item = /^\s*-\s+(.*)$/.exec(line);
    if (item && listKey) {
      const value = unquote(item[1]!);
      if (value) {
        if (listKey === "aliases") fm.aliases.push(value);
        else if (listKey === "tags") fm.tags.push(value);
      }
      continue;
    }

    const pair = /^([A-Za-z0-9_.-]+)\s*:\s*(.*)$/.exec(line);
    if (!pair) continue;
    const key = pair[1]!.toLowerCase();
    const value = pair[2]!;

    if (value.trim() === "") {
      // Une clé sans valeur ouvre potentiellement une liste sur les lignes suivantes.
      listKey = key === "aliases" || key === "alias" ? "aliases" : key === "tags" ? "tags" : null;
      continue;
    }
    listKey = null;

    if (key === "title") fm.title = unquote(value) || undefined;
    else if (key === "aliases" || key === "alias") fm.aliases.push(...parseInlineList(value));
    else if (key === "tags" || key === "tag") fm.tags.push(...parseInlineList(value));
    else {
      const scalar = unquote(value);
      // Les valeurs structurées (objets, blocs) ne sont pas du texte utile.
      if (scalar && !scalar.startsWith("{") && !scalar.startsWith("|") && !scalar.startsWith(">")) {
        fm.extra[key] = scalar;
      }
    }
  }

  return { frontmatter: fm, body };
}

/**
 * Le titre du frontmatter, s'il y en a un. `null` laisse l'appelant retomber
 * sur la règle d'origine de QMD (premier `#`, puis nom de fichier).
 */
export function frontmatterTitle(content: string): string | null {
  const { frontmatter } = parseFrontmatter(content);
  return frontmatter?.title?.trim() || null;
}

// =============================================================================
// Wikilinks (D-12, D-13)
// =============================================================================

export type Wikilink = {
  /** Cible telle qu'écrite, sans ancre ni extension : `carte/env/M3_REDIS_URL`. */
  target: string;
  /** Ancre `#section`, si présente. */
  anchor: string | null;
  /** Texte affiché quand il diffère de la cible. */
  alias: string | null;
  /** Vrai pour un `![[embed]]`. */
  embed: boolean;
};

// `[[cible#ancre|alias]]`, précédé éventuellement de `!` pour un embed.
const WIKILINK_RE = /(!?)\[\[([^\]\[|#]+)(#[^\]\[|]*)?(\|[^\]\[]*)?\]\]/g;

/** Retire l'extension `.md` d'une cible, Obsidian l'omettant par convention. */
function stripMd(target: string): string {
  return target.replace(/\.md$/i, "");
}

/** Tous les wikilinks d'un texte, dans l'ordre d'apparition. */
export function extractWikilinks(text: string): Wikilink[] {
  const links: Wikilink[] = [];
  for (const m of text.matchAll(WIKILINK_RE)) {
    const target = stripMd(m[2]!.trim());
    if (!target) continue;
    links.push({
      target,
      anchor: m[3] ? m[3].slice(1).trim() || null : null,
      alias: m[4] ? m[4].slice(1).trim() || null : null,
      embed: m[1] === "!",
    });
  }
  return links;
}

/** Le dernier segment d'un chemin : `carte/env/X` → `X`. */
export function shortName(target: string): string {
  const cut = target.lastIndexOf("/");
  return cut === -1 ? target : target.slice(cut + 1);
}

/**
 * Remplace chaque wikilink par un texte qui porte **à la fois** le libellé
 * affiché et le chemin cible, de sorte que BM25 comme l'embedding voient les
 * deux. Sans cela, chercher `M3_LLM_GATEWAY_URL` ne remonte que la note qui le
 * porte en titre, jamais les vingt notes qui la lient.
 *
 *   [[carte/env/X|X]]  →  X (→ carte/env/X)
 *   [[carte/env/X]]    →  X (→ carte/env/X)
 *   [[WikiWord]]       →  WikiWord
 *   [[note#Section]]   →  note (section Section)
 */
export function normalizeWikilinks(text: string): string {
  return text.replace(WIKILINK_RE, (_full, _bang, rawTarget, rawAnchor, rawAlias) => {
    const target = stripMd(String(rawTarget).trim());
    if (!target) return "";
    const alias = rawAlias ? String(rawAlias).slice(1).trim() : "";
    const anchor = rawAnchor ? String(rawAnchor).slice(1).trim() : "";
    const label = alias || shortName(target);

    let out = label;
    if (target !== label) out += ` (→ ${target})`;
    if (anchor) out += ` (section ${anchor})`;
    return out;
  });
}

// =============================================================================
// Version du pipeline de texte
// =============================================================================

/**
 * Entre dans l'empreinte d'embedding (`getEmbeddingFingerprint`). Le titre d'un
 * document est vectorisé avec son texte (`formatDocForEmbedding`), et lire le
 * `title:` du frontmatter change ce titre pour les documents qui en portent un.
 * Sans ce composant, QMD considérerait les anciens vecteurs comme à jour alors
 * qu'ils ont été calculés sur un autre titre.
 *
 * À incrémenter à chaque évolution de ce qui entre dans le texte vectorisé.
 */
export const OBSIDIAN_PIPELINE_VERSION = "1";

// =============================================================================
// Ce que ce module ne fait PAS, et pourquoi
// =============================================================================

/**
 * ⚠ `normalizeWikilinks` n'est **pas** appliqué au texte indexé, et ce n'est pas
 * un oubli. Les positions de chunk (`content_vectors.pos`) indexent le corps
 * ORIGINAL : `extractSnippet` s'en sert pour découper l'extrait et calculer le
 * numéro de ligne rendu à l'utilisateur (`store.ts:5269-5307`). Transformer le
 * texte avant de le découper décalerait toutes ces positions, et chaque
 * résultat afficherait le mauvais extrait à la mauvaise ligne.
 *
 * Le coût de ne pas normaliser est faible : un wikilink brut
 * `[[carte/env/X|X]]` contient déjà `carte/env/X` **et** `X` en clair, donc
 * BM25 les tokenise tous les deux. Il en va de même des `aliases:` du
 * frontmatter, présents dans le corps indexé. Ce que QMD ne savait pas faire,
 * et que ce module apporte, c'est **résoudre** ces liens en un graphe
 * interrogeable, et lire le `title:` du frontmatter.
 *
 * `normalizeWikilinks` sert donc à l'AFFICHAGE (la sortie de `qmd links`), là
 * où aucune position n'est en jeu.
 */

// =============================================================================
// Résolution d'une cible de wikilink vers un chemin de document
// =============================================================================

/**
 * Obsidian résout un lien par nom court quand il n'est pas ambigu, et par
 * chemin sinon. On rend les candidats par ordre de préférence, à confronter
 * aux chemins réellement indexés :
 *   1. le chemin exact avec `.md`
 *   2. le chemin exact tel quel (fichier sans extension)
 *   3. le nom court, à chercher en suffixe
 */
export function targetCandidates(target: string): { exact: string[]; suffix: string } {
  const clean = stripMd(target).replace(/^\.\//, "");
  return {
    exact: [`${clean}.md`, clean],
    suffix: `/${shortName(clean)}.md`,
  };
}
