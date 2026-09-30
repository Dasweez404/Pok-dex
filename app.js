// Pokédex caméra : détection 100 % dans le navigateur (CLIP zero-shot) + données PokéAPI.
import { AutoModel, AutoTokenizer, AutoProcessor, CLIPTextModelWithProjection, CLIPVisionModelWithProjection, RawImage }
  from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5';

const API = 'https://pokeapi.co/api/v2';
const MODEL = 'Xenova/clip-vit-base-patch32';
const DINO = 'Xenova/dinov2-small';   // très bon en recherche d'images similaires (complète CLIP)
const $ = (id) => document.getElementById(id);
const video = $('video'), canvas = $('canvas'), statusEl = $('status');
const scanBtn = $('scan'), autoBox = $('auto');
const setStatus = (t) => (statusEl.textContent = t);

let species = [];            // [{id, slug, fr}]
let labelEmbeds = null;      // Float32Array (n * dim), normalisés (texte)
let artCount = 0;            // artworks effectivement indexés
let artEmbeds = null;        // Float32Array (n * dim), normalisés (artworks officiels)
let dim = 512;
let tokenizer, processor, textModel, visionModel;
let dinoProcessor = null, dinoModel = null, dinoDim = 384;
let artDino = null;          // Float32Array (n * dinoDim), normalisés
let facing = 'environment', stream = null, busy = false;

/* ---------- IndexedDB (cache des embeddings de texte) ---------- */
const idb = (mode, fn) => new Promise((res, rej) => {
  const open = indexedDB.open('pokedex', 1);
  open.onupgradeneeded = () => open.result.createObjectStore('kv');
  open.onerror = () => rej(open.error);
  open.onsuccess = () => {
    const req = fn(open.result.transaction('kv', mode).objectStore('kv'));
    req.onsuccess = () => res(req.result);
    req.onerror = () => rej(req.error);
  };
});
const cacheGet = (k) => idb('readonly', (s) => s.get(k)).catch(() => null);
const cachePut = (k, v) => idb('readwrite', (s) => s.put(v, k)).catch(() => {});

/* ---------- Liste des espèces (+ noms français) ---------- */
async function loadSpecies() {
  const cached = await cacheGet('species-v1');
  if (cached) return cached;
  const list = await (await fetch(`${API}/pokemon-species?limit=2000`)).json();
  const out = list.results.map((r) => ({
    id: +r.url.match(/\/(\d+)\/?$/)[1], slug: r.name, fr: null,
  })).sort((a, b) => a.id - b.id);
  try {
    const g = await fetch('https://graphql.pokeapi.co/v1beta2', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ query: '{pokemonspeciesname(where:{language_id:{_eq:5}},limit:3000){pokemon_species_id name}}' }),
    }).then((r) => r.json());
    const map = new Map(g.data.pokemonspeciesname.map((n) => [n.pokemon_species_id, n.name]));
    out.forEach((s) => (s.fr = map.get(s.id) || null));
  } catch { /* les noms FR seront récupérés à l'affichage */ }
  await cachePut('species-v1', out);
  return out;
}

/* ---------- Modèle CLIP + embeddings des noms ---------- */
async function loadModel() {
  setStatus('Chargement du modèle de reconnaissance (1re fois : ~100 Mo)…');
  [tokenizer, processor, textModel, visionModel] = await Promise.all([
    AutoTokenizer.from_pretrained(MODEL),
    AutoProcessor.from_pretrained(MODEL),
    CLIPTextModelWithProjection.from_pretrained(MODEL),
    CLIPVisionModelWithProjection.from_pretrained(MODEL),
  ]);
  try {
    [dinoProcessor, dinoModel] = await Promise.all([AutoProcessor.from_pretrained(DINO), AutoModel.from_pretrained(DINO)]);
  } catch (e) { console.warn('DINOv2 indisponible, repli sur CLIP seul', e); dinoProcessor = dinoModel = null; }
  const key = `labels-v1-${species.length}`;
  const cached = await cacheGet(key);
  if (cached) { labelEmbeds = cached; return loadArtIndex(); }

  const all = new Float32Array(species.length * dim);
  const B = 32;
  for (let i = 0; i < species.length; i += B) {
    setStatus(`Préparation du Pokédex… ${Math.round((i / species.length) * 100)} % (une seule fois)`);
    const texts = species.slice(i, i + B).map((s) => `a photo of the Pokémon ${s.slug.replace(/-/g, ' ')}`);
    const inputs = tokenizer(texts, { padding: true, truncation: true });
    const { text_embeds } = await textModel(inputs);
    all.set(normalizeRows(text_embeds.data, texts.length), i * dim);
    await new Promise((r) => setTimeout(r));
  }
  labelEmbeds = all;
  await cachePut(key, all);
  await loadArtIndex();
}

/* ---------- Index visuel : embeddings des artworks officiels (une seule fois) ---------- */
const artUrl = (id) => `https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/${id}.png`;

// Dessine une image sur fond blanc (les PNG transparents deviendraient noirs sinon).
function flatten(source, sw, sh, sx = 0, sy = 0, cw = sw, ch = sh) {
  const c = document.createElement('canvas');
  c.width = c.height = 224;
  const g = c.getContext('2d');
  g.fillStyle = '#fff'; g.fillRect(0, 0, 224, 224);
  const k = Math.min(224 / cw, 224 / ch);
  g.drawImage(source, sx, sy, cw, ch, (224 - cw * k) / 2, (224 - ch * k) / 2, cw * k, ch * k);
  return c;
}
async function embedCanvas(c) {
  const blob = await new Promise((r) => c.toBlob(r, 'image/jpeg', 0.92));
  const image = await RawImage.fromBlob(blob);
  const { pixel_values } = await processor(image);
  const { image_embeds } = await visionModel({ pixel_values });
  const out = { clip: normalizeRows(image_embeds.data, 1), dino: null };
  if (dinoModel) {
    const px = (await dinoProcessor(image)).pixel_values;
    const o = await dinoModel({ pixel_values: px });
    const v = o.pooler_output ? o.pooler_output.data : o.last_hidden_state.data.slice(0, dinoDim);
    out.dino = normalizeRows(v, 1, dinoDim);
  }
  return out;
}

async function loadArtIndex() {
  const key = `arts-v2-${species.length}-${dinoModel ? 'd' : 'c'}`;
  const cached = await cacheGet(key);
  if (cached) { artEmbeds = cached.clip; artDino = cached.dino; artCount = species.length; return; }
  const n = species.length, all = new Float32Array(n * dim), allD = new Float32Array(n * dinoDim), done = new Uint8Array(n);
  const B = 8;
  for (let i = 0; i < n; i += B) {
    setStatus(`Apprentissage des artworks officiels… ${Math.round((i / n) * 100)} % (une seule fois, garde la page ouverte)`);
    await Promise.all(species.slice(i, i + B).map(async (sp, j) => {
      for (let t = 0; t < 3 && !done[i + j]; t++) {
        try {
          const r = await fetch(artUrl(sp.id));
          if (!r.ok) throw new Error(r.status);
          const bmp = await createImageBitmap(await r.blob());
          const e = await embedCanvas(flatten(bmp, bmp.width, bmp.height));
          all.set(e.clip, (i + j) * dim);
          if (e.dino) allD.set(e.dino, (i + j) * dinoDim);
          done[i + j] = 1;
        } catch { await new Promise((r) => setTimeout(r, 400 * (t + 1))); }
      }
    }));
  }
  artEmbeds = all; artDino = dinoModel ? allD : null;
  artCount = done.reduce((a, b) => a + b, 0);
  if (artCount > n * 0.98) await cachePut(key, { clip: all, dino: artDino });
}

function normalizeRows(data, rows, dm = dim) {
  const out = new Float32Array(rows * dm);
  for (let r = 0; r < rows; r++) {
    let n = 0;
    for (let d = 0; d < dm; d++) n += data[r * dm + d] ** 2;
    n = Math.sqrt(n) || 1;
    for (let d = 0; d < dm; d++) out[r * dm + d] = data[r * dm + d] / n;
  }
  return out;
}

async function classify() {
  const w = video.videoWidth, h = video.videoHeight;
  const side = Math.min(w, h);
  // Trois cadrages (le Pokémon peut être petit dans l'image) : centre 50 %, 70 % et 100 %.
  const crops = [0.5, 0.7, 1].map((k) =>
    flatten(video, w, h, (w - side * k) / 2, (h - side * k) / 2, side * k, side * k));
  const vs = [];
  for (const c of crops) vs.push(await embedCanvas(c));

  const n = species.length, logits = new Float32Array(n);
  let bestImg = 0;
  for (let i = 0; i < n; i++) {
    let img = 0, txt = 0, dn = 0;
    for (const v of vs) {
      let a = 0, t = 0;
      for (let d = 0; d < dim; d++) { a += v.clip[d] * artEmbeds[i * dim + d]; t += v.clip[d] * labelEmbeds[i * dim + d]; }
      img = Math.max(img, a); txt = Math.max(txt, t);
      if (v.dino && artDino) {
        let q = 0;
        for (let d = 0; d < dinoDim; d++) q += v.dino[d] * artDino[i * dinoDim + d];
        dn = Math.max(dn, q);
      }
    }
    bestImg = Math.max(bestImg, img);
    // DINOv2 (forme/apparence précise) > CLIP image > texte (départage)
    logits[i] = artDino ? dn * 140 + img * 50 + txt * 30 : img * 100 + txt * 50;
  }
  const max = Math.max(...logits);
  const exps = Array.from(logits, (s) => Math.exp(s - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  const top = exps.map((e, i) => ({ sp: species[i], p: e / sum })).sort((a, b) => b.p - a.p).slice(0, 4);
  top.bestImg = bestImg;
  return top;
}

/* ---------- Données Pokémon (PokéAPI) ---------- */
const clean = (t) => t.replace(/[\n\f\r­]+/g, ' ').replace(/\s+/g, ' ').trim();

function mergeEntries(entries, lang) {
  const seen = new Set(), parts = [];
  for (const e of entries) {
    if (e.language.name !== lang) continue;
    const t = clean(e.flavor_text);
    const k = t.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, '');
    if (!k || seen.has(k)) continue;
    seen.add(k);
    parts.push(/[.!?…»)]$/.test(t) ? t : t + '.');
  }
  return parts.join(' ');
}

const TYPE_COLORS = {
  normal: '#6d6d5a', fire: '#c2410c', water: '#2563eb', grass: '#2f7d3b', electric: '#a16207', ice: '#0e7490',
  fighting: '#b91c1c', poison: '#7e3fa0', ground: '#92602a', flying: '#4f5fb8', psychic: '#c2315e', bug: '#5f7d12',
  rock: '#7a6a3a', ghost: '#5b4a8a', dragon: '#4c3fc0', dark: '#3f3a38', steel: '#4b6a7a', fairy: '#b5467e',
};
const FALLBACK_COLOR = '#4b5563';
const REGIONS = { i: 'Kanto', ii: 'Johto', iii: 'Hoenn', iv: 'Sinnoh', v: 'Unys', vi: 'Kalos', vii: 'Alola', viii: 'Galar', ix: 'Paldea' };
const DEX_REGIONS = [
  ['kanto', 'Kanto'], ['johto', 'Johto'], ['hoenn', 'Hoenn'], ['sinnoh', 'Sinnoh'], ['unova', 'Unys'],
  ['kalos', 'Kalos'], ['alola', 'Alola'], ['galar', 'Galar'], ['isle-of-armor', 'Galar'], ['crown-tundra', 'Galar'],
  ['hisui', 'Hisui'], ['paldea', 'Paldea'], ['kitakami', 'Paldea'], ['blueberry', 'Paldea'],
];
const STAT_LABELS = { hp: 'PV', attack: 'Attaque', defense: 'Défense', 'special-attack': 'Att. Spé.', 'special-defense': 'Déf. Spé.', speed: 'Vitesse' };

const getJson = async (url) => (await fetch(url)).json();
const frName = (o, fallback) => o.names?.find((n) => n.language.name === 'fr')?.name || fallback;

async function loadPokemon(id) {
  const sp = await getJson(`${API}/pokemon-species/${id}`);
  const fr = frName(sp, sp.name);
  const en = sp.names.find((n) => n.language.name === 'en')?.name || sp.name;
  let desc = mergeEntries(sp.flavor_text_entries, 'fr'), lang = 'fr';
  if (!desc) { desc = mergeEntries(sp.flavor_text_entries, 'en'); lang = 'en'; }
  const genus = (sp.genera.find((g) => g.language.name === 'fr') || sp.genera.find((g) => g.language.name === 'en'))?.genus || '';

  const p = { id: sp.id, fr, en, desc, lang, genus, types: [], abilities: [], stats: [], habitat: '', origin: '', regions: [], height: null, weight: null };
  p.origin = REGIONS[sp.generation?.name?.replace('generation-', '')] || '';
  const dex = new Set();
  for (const d of sp.pokedex_numbers) {
    const hit = DEX_REGIONS.find(([k]) => d.pokedex.name.includes(k));
    if (hit) dex.add(hit[1]);
  }
  p.regions = [...dex];

  const tasks = [];
  const variety = sp.varieties.find((v) => v.is_default) || sp.varieties[0];
  tasks.push(getJson(variety.pokemon.url).then(async (poke) => {
    p.height = poke.height / 10; p.weight = poke.weight / 10;
    p.stats = poke.stats.map((s) => ({ label: STAT_LABELS[s.stat.name] || s.stat.name, value: s.base_stat }));
    const [types, abilities] = await Promise.all([
      Promise.all(poke.types.map(async (t) => ({ slug: t.type.name, name: frName(await getJson(t.type.url), t.type.name) }))),
      Promise.all(poke.abilities.map(async (a) => ({ name: frName(await getJson(a.ability.url), a.ability.name), hidden: a.is_hidden }))),
    ]);
    p.types = types; p.abilities = abilities;
  }).catch(() => {}));
  if (sp.habitat) tasks.push(getJson(sp.habitat.url).then((h) => { p.habitat = frName(h, sp.habitat.name); }).catch(() => {}));
  await Promise.all(tasks);
  return p;
}

const el = (tag, props = {}, ...kids) => { const e = Object.assign(document.createElement(tag), props); e.append(...kids); return e; };
const fmt = (n) => n.toLocaleString('fr-FR', { maximumFractionDigits: 1 });

let alternatives = [];
async function show(id, alts = []) {
  setStatus('Recherche des informations…');
  try {
    const p = await loadPokemon(id);
    const color = TYPE_COLORS[p.types[0]?.slug] || FALLBACK_COLOR;
    document.documentElement.style.setProperty('--t', color);
    document.querySelector('meta[name=theme-color]').content = color;
    $('art').src = `https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/${p.id}.png`;
    $('art').alt = p.fr;
    $('num').textContent = `N° ${String(p.id).padStart(4, '0')}`;
    $('fr').textContent = p.fr;
    $('en').textContent = [p.en !== p.fr ? p.en : '', p.genus].filter(Boolean).join(' · ');
    $('types').replaceChildren(...p.types.map((t) => el('span', { textContent: t.name, style: `color:${TYPE_COLORS[t.slug] || FALLBACK_COLOR}` })));
    $('desc').textContent = p.desc || 'Aucune entrée de Pokédex disponible.';
    $('langnote').textContent = p.lang === 'en' ? 'Entrées disponibles uniquement en anglais.' : '';

    const facts = [
      ['Taille', p.height != null && `${fmt(p.height)} m`],
      ['Poids', p.weight != null && `${fmt(p.weight)} kg`],
      ['Région d’origine', p.origin],
      ['Habitat', p.habitat],
      ['Talents', p.abilities.length && p.abilities.map((a) => a.hidden ? `${a.name} (caché)` : a.name).join(', ')],
      ['Présent dans les Pokédex de', p.regions.length && p.regions.join(', ')],
    ].filter(([, v]) => v);
    $('facts').replaceChildren(...facts.map(([k, v]) => el('div', {}, el('dt', { textContent: k }), el('dd', { textContent: v }))));

    $('stats').replaceChildren(...p.stats.map((s) => el('div', { className: 'stat' },
      el('span', { textContent: s.label }), el('b', { textContent: s.value }),
      el('i', {}, el('u', { style: `width:${Math.min(100, (s.value / 200) * 100)}%` })))));

    const box = $('alts');
    box.replaceChildren();
    if (alts.length) {
      box.append(el('small', { textContent: 'Ce n’est pas lui ? ' }));
      for (const a of alts) {
        box.append(el('button', {
          textContent: `${a.sp.fr || a.sp.slug} (${Math.round(a.p * 100)} %)`,
          onclick: () => show(a.sp.id, alts.filter((x) => x !== a)),
        }));
      }
    }
    $('result').hidden = false;
    $('result').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    setStatus(`Détecté : ${p.fr}${artCount < species.length * 0.98 ? ` — ⚠ index visuel incomplet (${artCount}/${species.length})` : ''}`);
  } catch (e) {
    setStatus('Erreur réseau : impossible de charger les données.');
    console.error(e);
  }
}

/* ---------- Caméra & scan ---------- */
async function startCamera() {
  stream?.getTracks().forEach((t) => t.stop());
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: facing }, audio: false });
    video.srcObject = stream;
  } catch {
    setStatus('Caméra indisponible : utilisez la recherche par nom.');
  }
}

let lastId = null;
async function scan() {
  if (busy || !labelEmbeds || !artEmbeds || !video.videoWidth) return;
  busy = true; scanBtn.disabled = true;
  try {
    setStatus('Analyse…');
    const top = await classify();
    if (top.bestImg < 0.45) { setStatus(`Pas de Pokémon reconnu — rapprochez-vous / centrez-le.`); return; }
    if (autoBox.checked && top[0].sp.id === lastId) { setStatus(`Détecté : ${top[0].sp.fr || top[0].sp.slug}`); return; }
    lastId = top[0].sp.id;
    await show(top[0].sp.id, top.slice(1, 4));
  } catch (e) { console.error(e); setStatus('Erreur pendant l’analyse.'); }
  finally { busy = false; scanBtn.disabled = false; }
}

scanBtn.onclick = scan;
$('flip').onclick = () => { facing = facing === 'environment' ? 'user' : 'environment'; startCamera(); };
setInterval(() => { if (autoBox.checked) scan(); }, 2000);

$('search').onsubmit = (e) => {
  e.preventDefault();
  const q = $('q').value.trim().toLowerCase();
  const norm = (s) => (s || '').normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase();
  const s = species.find((x) => String(x.id) === q || norm(x.fr) === norm(q) || x.slug === norm(q));
  if (s) show(s.id); else setStatus('Pokémon introuvable.');
};

(async function init() {
  startCamera();
  try {
    species = await loadSpecies();
    $('names').replaceChildren(...species.filter((s) => s.fr).map((s) => Object.assign(document.createElement('option'), { value: s.fr })));
    await loadModel();
    scanBtn.disabled = false;
    setStatus('Prêt : cadrez un Pokémon (peluche, carte, écran…) et appuyez sur Scanner.');
  } catch (e) {
    console.error(e);
    setStatus('Échec du chargement (réseau ?). La recherche par nom reste disponible.');
  }
})();
