// Pokédex caméra : détection 100 % dans le navigateur (CLIP zero-shot) + données PokéAPI.
import { AutoTokenizer, AutoProcessor, CLIPTextModelWithProjection, CLIPVisionModelWithProjection, RawImage }
  from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5';

const API = 'https://pokeapi.co/api/v2';
const MODEL = 'Xenova/clip-vit-base-patch32';
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
  const { pixel_values } = await processor(await RawImage.fromBlob(blob));
  const { image_embeds } = await visionModel({ pixel_values });
  return normalizeRows(image_embeds.data, 1);
}

async function loadArtIndex() {
  const key = `arts-v1-${species.length}`;
  const cached = await cacheGet(key);
  if (cached) { artEmbeds = cached; artCount = species.length; return; }
  const n = species.length, all = new Float32Array(n * dim), done = new Uint8Array(n);
  const B = 8;
  for (let i = 0; i < n; i += B) {
    setStatus(`Apprentissage des artworks officiels… ${Math.round((i / n) * 100)} % (une seule fois, garde la page ouverte)`);
    await Promise.all(species.slice(i, i + B).map(async (sp, j) => {
      for (let t = 0; t < 3 && !done[i + j]; t++) {
        try {
          const r = await fetch(artUrl(sp.id));
          if (!r.ok) throw new Error(r.status);
          const bmp = await createImageBitmap(await r.blob());
          all.set(await embedCanvas(flatten(bmp, bmp.width, bmp.height)), (i + j) * dim);
          done[i + j] = 1;
        } catch { await new Promise((r) => setTimeout(r, 400 * (t + 1))); }
      }
    }));
  }
  // Une espèce sans artwork garde un vecteur nul : son score image sera 0.
  artEmbeds = all;
  artCount = done.reduce((a, b) => a + b, 0);
  if (artCount > n * 0.98) await cachePut(key, all);
}

function normalizeRows(data, rows) {
  const out = new Float32Array(rows * dim);
  for (let r = 0; r < rows; r++) {
    let n = 0;
    for (let d = 0; d < dim; d++) n += data[r * dim + d] ** 2;
    n = Math.sqrt(n) || 1;
    for (let d = 0; d < dim; d++) out[r * dim + d] = data[r * dim + d] / n;
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
  const imgs = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let img = 0, txt = 0;
    for (const v of vs) {
      let a = 0, t = 0;
      for (let d = 0; d < dim; d++) { a += v[d] * artEmbeds[i * dim + d]; t += v[d] * labelEmbeds[i * dim + d]; }
      img = Math.max(img, a); txt = Math.max(txt, t);
    }
    bestImg = Math.max(bestImg, img);
    imgs[i] = img;
    logits[i] = img * 100 + txt * 50;          // l'image prime, le texte départage
  }
  const max = Math.max(...logits);
  const exps = Array.from(logits, (s) => Math.exp(s - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  const top = exps.map((e, i) => ({ sp: species[i], p: e / sum })).sort((a, b) => b.p - a.p).slice(0, 4);
  top.bestImg = bestImg;
  const order = top.map((t) => t.sp.id);
  top.sims = order.map((id) => imgs[species.findIndex((x) => x.id === id)]);
  top.dbg = `[debug] index ${artCount}/${n} · sim. ${top.map((t, i) => `${t.sp.slug} ${top.sims[i].toFixed(2)}`).join(' | ')}`;
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

async function loadPokemon(id) {
  const sp = await (await fetch(`${API}/pokemon-species/${id}`)).json();
  const fr = sp.names.find((n) => n.language.name === 'fr')?.name || sp.name;
  const en = sp.names.find((n) => n.language.name === 'en')?.name || sp.name;
  let desc = mergeEntries(sp.flavor_text_entries, 'fr'), lang = 'fr';
  if (!desc) { desc = mergeEntries(sp.flavor_text_entries, 'en'); lang = 'en'; }

  let types = [];
  try {
    const variety = sp.varieties.find((v) => v.is_default) || sp.varieties[0];
    const poke = await (await fetch(variety.pokemon.url)).json();
    types = await Promise.all(poke.types.map(async (t) => {
      const ty = await (await fetch(t.type.url)).json();
      return ty.names.find((n) => n.language.name === 'fr')?.name || t.type.name;
    }));
  } catch { /* types facultatifs */ }
  return { id: sp.id, fr, en, desc, lang, types };
}

let alternatives = [];
async function show(id, alts = []) {
  setStatus('Recherche des informations…');
  try {
    const p = await loadPokemon(id);
    $('art').src = `https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/${p.id}.png`;
    $('art').alt = p.fr;
    $('num').textContent = `N° ${String(p.id).padStart(4, '0')}`;
    $('fr').textContent = p.fr;
    $('en').textContent = p.en !== p.fr ? p.en : '';
    $('types').replaceChildren(...p.types.map((t) => Object.assign(document.createElement('span'), { textContent: t })));
    $('desc').textContent = p.desc || 'Aucune entrée de Pokédex disponible.';
    $('langnote').textContent = p.lang === 'en' ? 'Entrées disponibles uniquement en anglais.' : '';
    const box = $('alts');
    box.replaceChildren();
    if (alts.length) {
      box.append(Object.assign(document.createElement('small'), { textContent: 'Ce n’est pas lui ? ' }));
      for (const a of alts) {
        const b = document.createElement('button');
        b.textContent = `${a.sp.fr || a.sp.slug} (${Math.round(a.p * 100)} %)`;
        b.onclick = () => show(a.sp.id, alts.filter((x) => x !== a));
        box.append(b);
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
    if (top.bestImg < 0.45) { setStatus(`Pas de Pokémon reconnu — rapprochez-vous / centrez-le. ${top.dbg}`); return; }
    if (autoBox.checked && top[0].sp.id === lastId) { setStatus(`Détecté : ${top[0].sp.fr || top[0].sp.slug}`); return; }
    lastId = top[0].sp.id;
    console.log(top.dbg, top.map((t) => `${t.sp.slug} ${t.p.toFixed(2)}`));
    await show(top[0].sp.id, top.slice(1, 4));
    $('langnote').textContent += ` ${top.dbg}`;
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
