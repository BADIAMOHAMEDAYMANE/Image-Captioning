// Génère presentation.pptx — deck du projet Image Captioning (français).
// Usage:  NODE_PATH=/tmp/node_modules node make_slides.js
const pptxgen = require("pptxgenjs");

const P = "065A82";   // bleu profond (dominant)
const T = "1C7293";   // teal (secondaire)
const M = "21295C";   // midnight (fonds sombres)
const ICE = "E8F1F6"; // fond clair de cartes
const INK = "20303C"; // texte principal
const MUT = "5E7484"; // texte atténué
const WHT = "FFFFFF";
const OUT = __dirname + "/../outputs/";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";                    // 13.33 x 7.5
pres.defineSlideMaster({ title: "LIGHT", background: { color: WHT } });
pres.defineSlideMaster({ title: "DARK", background: { color: M } });

// ---------- helpers ----------
// motif du projet : grille d'attention 3x3, une case "allumée"
function attnGrid(s, x, y, cell, lit, dark) {
  for (let r = 0; r < 3; r++)
    for (let c = 0; c < 3; c++) {
      const isLit = r * 3 + c === lit;
      s.addShape("roundRect", {
        x: x + c * (cell + 0.06), y: y + r * (cell + 0.06),
        w: cell, h: cell, rectRadius: 0.03,
        fill: { color: isLit ? "02C39A" : dark ? "3A4E8C" : ICE },
        line: { color: dark ? "55679D" : "C9D9E4", width: 0.75 },
      });
    }
}
function title(s, txt, dark) {
  s.addText(txt, { x: 0.6, y: 0.35, w: 10.6, h: 0.75, fontFace: "Cambria",
    fontSize: 30, bold: true, color: dark ? WHT : M, margin: 0 });
  attnGrid(s, 12.05, 0.38, 0.20, Math.floor(Math.random() * 9), dark);
}
function bullets(s, items, opts) {
  s.addText(items.map((t, i) => ({
    text: t.t, options: { bullet: !t.plain, bold: !!t.b, breakLine: true,
      color: t.c || INK, fontSize: t.s || 15, paraSpaceAfter: 8 } })),
    Object.assign({ fontFace: "Calibri", valign: "top", margin: 0 }, opts));
}
function box(s, x, y, w, h, txt, fill, color, size, bold, extra) {
  s.addShape("roundRect", { x, y, w, h, rectRadius: 0.06,
    fill: { color: fill }, line: { color: "C9D9E4", width: 0.75 } });
  s.addText(txt, Object.assign({ x, y, w, h, align: "center",
    valign: "middle", fontFace: "Calibri", fontSize: size, bold: !!bold,
    color, margin: 0.04 }, extra || {}));
}
function arrowR(s, x, y, w) {
  s.addShape("rightArrow", { x, y, w, h: 0.28,
    fill: { color: T }, line: { type: "none" } });
}
function arrowD(s, x, y, h) {
  s.addShape("downArrow", { x, y, w: 0.28, h,
    fill: { color: T }, line: { type: "none" } });
}
function note(s, txt) { s.addNotes(txt); }

// ============ S1 — Titre (sombre) ============
let s = pres.addSlide({ masterName: "DARK" });
attnGrid(s, 9.6, 1.9, 1.05, 4, true);
s.addText("Image Captioning", { x: 0.9, y: 2.2, w: 8.2, h: 1.1,
  fontFace: "Cambria", fontSize: 48, bold: true, color: WHT, margin: 0 });
s.addText("De CNN + LSTM au Transformer — un projet from scratch en PyTorch",
  { x: 0.9, y: 3.35, w: 8.2, h: 0.6, fontFace: "Calibri", fontSize: 19,
    color: "A9CCE0", margin: 0 });
s.addText("Mohamed Aymane", { x: 0.9, y: 4.35, w: 8.2, h: 0.4,
  fontFace: "Calibri", fontSize: 15, bold: true, color: WHT, margin: 0 });
s.addText("Trois décodeurs · un seul pipeline · comparaison complète",
  { x: 0.9, y: 6.6, w: 11.5, h: 0.4, fontFace: "Calibri", fontSize: 13,
    italic: true, color: "7E9BC8", margin: 0 });
note(s, "Objectif : construire, comprendre et comparer trois architectures de génération de descriptions d'images.");

// ============ S2 — Le problème ============
s = pres.addSlide({ masterName: "LIGHT" });
title(s, "Le problème : faire parler une image");
bullets(s, [
  { t: "Un problème multimodal", b: true, s: 17, plain: true },
  { t: "L'entrée est une image, la sortie est du texte : le modèle doit relier le monde visuel et le monde du langage." },
  { t: "Voir — reconnaître les objets (chien, plage, eau)." },
  { t: "Comprendre — saisir les relations (le chien court sur le sable)." },
  { t: "Parler — produire une phrase correcte, mot par mot." },
], { x: 0.6, y: 1.5, w: 6.4, h: 4.4 });
// mini-flux vertical à droite
box(s, 8.2, 1.5, 3.6, 0.75, "Image  (pixels)", ICE, INK, 15, true);
arrowD(s, 9.86, 2.32, 0.5);
box(s, 8.2, 2.9, 3.6, 0.75, "Encodeur — « les yeux »", P, WHT, 15, true);
arrowD(s, 9.86, 3.72, 0.5);
box(s, 8.2, 4.3, 3.6, 0.75, "Décodeur — « la bouche »", P, WHT, 15, true);
arrowD(s, 9.86, 5.12, 0.5);
box(s, 8.2, 5.7, 3.6, 0.75, "« a dog runs on the beach »", ICE, INK, 14, false, { italic: true });
note(s, "Analogie : décrire une photo au téléphone — les yeux voient, la bouche décrit.");

// ============ S3 — Applications ============
s = pres.addSlide({ masterName: "LIGHT" });
title(s, "Pourquoi c'est utile");
const apps = [
  ["Accessibilité", "Décrire les photos aux personnes malvoyantes (alt-text automatique)."],
  ["Conduite autonome", "Expliquer la scène en mots : « un piéton traverse à gauche »."],
  ["Santé", "Brouillons de rapports pour radiographies, relus par le médecin."],
  ["Recherche & e-commerce", "Indexer les images par leur contenu, décrire les produits."],
];
apps.forEach((a, i) => {
  const x = 0.6 + (i % 2) * 6.25, y = 1.6 + Math.floor(i / 2) * 2.65;
  s.addShape("roundRect", { x, y, w: 5.85, h: 2.3, rectRadius: 0.08,
    fill: { color: ICE }, line: { color: "C9D9E4", width: 0.75 } });
  s.addShape("ellipse", { x: x + 0.35, y: y + 0.4, w: 0.62, h: 0.62,
    fill: { color: P }, line: { type: "none" } });
  s.addText(String(i + 1), { x: x + 0.35, y: y + 0.4, w: 0.62, h: 0.62,
    align: "center", valign: "middle", fontFace: "Calibri", fontSize: 18,
    bold: true, color: WHT, margin: 0 });
  s.addText(a[0], { x: x + 1.2, y: y + 0.38, w: 4.4, h: 0.55,
    fontFace: "Calibri", fontSize: 18, bold: true, color: M, margin: 0 });
  s.addText(a[1], { x: x + 1.2, y: y + 1.0, w: 4.35, h: 1.1,
    fontFace: "Calibri", fontSize: 13.5, color: INK, margin: 0 });
});
note(s, "Quatre familles d'applications concrètes.");

// ============ S4 — Pipeline ============
s = pres.addSlide({ masterName: "LIGHT" });
title(s, "Le pipeline du projet");
const steps = [
  ["1. Données", "nettoyage · vocabulaire\nsplits par image", "prepare_data.py"],
  ["2. Features", "ResNet50 gelé\nprécalculées une fois", "extract_features.py"],
  ["3. Entraînement", "teacher forcing\nearly stopping · resume", "train.py --model X"],
  ["4. Résultats", "BLEU · captions\nheatmaps · tableaux", "evaluate.py · inference.py"],
];
steps.forEach((st, i) => {
  const x = 0.6 + i * 3.22;
  box(s, x, 2.1, 2.75, 1.9, "", ICE, INK, 13);
  s.addText(st[0], { x: x + 0.15, y: 2.25, w: 2.45, h: 0.45,
    fontFace: "Calibri", fontSize: 17, bold: true, color: P, margin: 0 });
  s.addText(st[1], { x: x + 0.15, y: 2.75, w: 2.45, h: 1.0,
    fontFace: "Calibri", fontSize: 13, color: INK, margin: 0 });
  s.addText(st[2], { x: x + 0.15, y: 4.15, w: 2.45, h: 0.4,
    fontFace: "Courier New", fontSize: 11.5, color: MUT, margin: 0 });
  if (i < 3) arrowR(s, x + 2.8, 2.9, 0.38);
});
s.addText("Décision clé : l'encodeur est gelé, donc les features sont précalculées → epochs ~10× plus rapides et comparaison équitable des trois décodeurs sur les mêmes données.",
  { x: 0.6, y: 5.3, w: 12.1, h: 0.9, fontFace: "Calibri", fontSize: 14.5,
    italic: true, color: MUT, margin: 0 });
note(s, "Quatre étapes, quatre scripts. Les features précalculées sont la décision de design la plus rentable.");

// ============ S5 — Données ============
s = pres.addSlide({ masterName: "LIGHT" });
title(s, "Données : Flickr8k");
const stats = [["8 091", "images"], ["5", "captions par image"], ["≈ 3 000", "mots de vocabulaire"]];
stats.forEach((st, i) => {
  const x = 0.6 + i * 4.15;
  s.addShape("roundRect", { x, y: 1.6, w: 3.7, h: 1.75, rectRadius: 0.08,
    fill: { color: i === 1 ? P : ICE }, line: { color: "C9D9E4", width: 0.75 } });
  s.addText(st[0], { x, y: 1.75, w: 3.7, h: 0.9, align: "center",
    fontFace: "Cambria", fontSize: 40, bold: true,
    color: i === 1 ? WHT : P, margin: 0 });
  s.addText(st[1], { x, y: 2.7, w: 3.7, h: 0.45, align: "center",
    fontFace: "Calibri", fontSize: 14, color: i === 1 ? "CADCFC" : MUT, margin: 0 });
});
bullets(s, [
  { t: "Préparation du texte", b: true, s: 16, plain: true },
  { t: "Minuscules, ponctuation retirée, tokens spéciaux <start> <end> <pad> <unk>." },
  { t: "Vocabulaire construit sur le train uniquement (pas de fuite), fréquence ≥ 5." },
  { t: "Split 80 / 10 / 10 par image — les 5 captions d'une image restent ensemble." },
  { t: "Padding par batch et loss qui ignore <pad> (le bug classique évité)." },
], { x: 0.6, y: 3.9, w: 12.0, h: 3.0 });
note(s, "Flickr8k : assez petit pour itérer vite, assez riche pour apprendre.");

// ============ S6 — Architecture 1 ============
s = pres.addSlide({ masterName: "LIGHT" });
title(s, "Architecture 1 — CNN + LSTM (baseline)");
bullets(s, [
  { t: "ResNet50 pré-entraîné et gelé → un vecteur global de 2048 nombres." },
  { t: "Ce vecteur initialise l'état caché (h₀, c₀) du LSTM." },
  { t: "Génération mot par mot avec teacher forcing à l'entraînement." },
  { t: "Inférence : greedy search ou beam search (normalisé par longueur)." },
  { t: "Limite : toute l'image doit tenir dans un seul vecteur.", b: true },
], { x: 0.6, y: 1.6, w: 6.6, h: 4.4 });
box(s, 8.0, 1.6, 4.4, 0.7, "Image 224 × 224", ICE, INK, 14, true);
arrowD(s, 10.06, 2.37, 0.4);
box(s, 8.0, 2.82, 4.4, 0.7, "ResNet50 (gelé)", P, WHT, 14, true);
arrowD(s, 10.06, 3.59, 0.4);
box(s, 8.0, 4.04, 4.4, 0.7, "vecteur global (2048)", ICE, INK, 14);
arrowD(s, 10.06, 4.81, 0.4);
box(s, 8.0, 5.26, 4.4, 0.7, "LSTM → « a dog runs … »", P, WHT, 14, true);
note(s, "Le modèle 'Show and Tell' (2015). Simple, rapide, mais goulot d'étranglement.");

// ============ S7 — Architecture 2 ============
s = pres.addSlide({ masterName: "LIGHT" });
title(s, "Architecture 2 — Attention de Bahdanau");
bullets(s, [
  { t: "L'image devient 49 régions (grille 7×7), chacune décrite par 2048 nombres. À chaque mot, le décodeur calcule un poids par région et « regarde » là où il faut." },
  { t: "Régularisation doublement stochastique : l'attention doit couvrir toute l'image. Bonus : les poids se visualisent en heatmaps — le modèle devient interprétable." },
], { x: 0.6, y: 1.35, w: 12.1, h: 1.6 });
s.addImage({ path: OUT + "attention_dummy_038.png",
  x: 1.3, y: 3.05, w: 10.7, h: 4.0 });
s.addText("Exemple du pipeline de test (encodeur aléatoire) — à régénérer sur Flickr8k pour des heatmaps focalisées.",
  { x: 1.3, y: 7.05, w: 10.7, h: 0.35, align: "center", fontFace: "Calibri",
    fontSize: 11, italic: true, color: MUT, margin: 0 });
note(s, "'Show, Attend and Tell' (2015). Une heatmap par mot généré : où regarde le modèle quand il dit 'circle' ?");

// ============ S8 — Architecture 3 ============
s = pres.addSlide({ masterName: "LIGHT" });
title(s, "Architecture 3 — Décodeur Transformer");
bullets(s, [
  { t: "Self-attention masquée : chaque mot voit tous les mots précédents — directement." },
  { t: "Cross-attention : les mots interrogent les 49 régions de l'image à chaque couche." },
  { t: "Positional encoding sinusoïdal : l'ordre des mots sans récurrence." },
  { t: "Entraînement parallèle sur tous les pas de temps → epochs rapides." },
  { t: "3 couches, 8 têtes, d_model 512 — via nn.TransformerDecoder." },
], { x: 0.6, y: 1.6, w: 6.6, h: 4.6 });
box(s, 8.0, 1.6, 4.4, 0.65, "mots générés + PE", ICE, INK, 13, true);
arrowD(s, 10.06, 2.32, 0.33);
box(s, 8.0, 2.7, 4.4, 0.85, "self-attention masquée", P, WHT, 14, true);
arrowD(s, 10.06, 3.62, 0.33);
box(s, 8.0, 4.0, 4.4, 0.85, "cross-attention → image (49 régions)", T, WHT, 14, true);
arrowD(s, 10.06, 4.92, 0.33);
box(s, 8.0, 5.3, 4.4, 0.65, "feed-forward  (× 3 couches)", ICE, INK, 13, true);
s.addText("Mémoire image normalisée (LayerNorm) — sinon la cross-attention sature.",
  { x: 8.0, y: 6.15, w: 4.4, h: 0.6, fontFace: "Calibri", fontSize: 11.5,
    italic: true, color: MUT, margin: 0 });
note(s, "Le passage du séquentiel au parallèle. LayerNorm sur la mémoire : bug réel trouvé pendant le projet.");

// ============ S9 — Entraînement ============
s = pres.addSlide({ masterName: "LIGHT" });
title(s, "Entraînement");
bullets(s, [
  { t: "Teacher forcing — le vrai mot précédent en entrée, apprentissage stable." },
  { t: "Cross-entropy qui ignore <pad> ; gradient clipping ; AdamW + ReduceLROnPlateau." },
], { x: 0.6, y: 1.4, w: 6.0, h: 1.9 });
bullets(s, [
  { t: "Early stopping (patience 5) — on garde le meilleur checkpoint, pas le dernier." },
  { t: "Checkpoints atomiques + --resume : un entraînement interrompu reprend proprement." },
], { x: 6.9, y: 1.4, w: 5.8, h: 1.9 });
s.addImage({ path: OUT + "training_curves.png", x: 0.75, y: 3.55, w: 11.8, h: 3.15 });
s.addText("Courbes du pipeline de test — les vraies courbes Flickr8k se génèrent avec plot_curves.py.",
  { x: 0.75, y: 6.75, w: 11.8, h: 0.35, align: "center", fontFace: "Calibri",
    fontSize: 11, italic: true, color: MUT, margin: 0 });
note(s, "Trois modèles, un seul script d'entraînement. Les courbes montrent la convergence et le point d'early stopping.");

// ============ S10 — Évaluation ============
s = pres.addSlide({ masterName: "LIGHT" });
title(s, "Évaluation : comparer une phrase à 5 références");
s.addTable([
  [{ text: "Métrique", options: { bold: true, color: WHT, fill: { color: P } } },
   { text: "Idée", options: { bold: true, color: WHT, fill: { color: P } } },
   { text: "Quand l'utiliser", options: { bold: true, color: WHT, fill: { color: P } } }],
  ["BLEU-1…4", "précision des n-grammes + pénalité de brièveté", "standard, comparable aux papiers"],
  ["METEOR", "synonymes et racines, précision + rappel", "crédit au sens, pas juste aux mots"],
  ["CIDEr", "n-grammes pondérés TF-IDF (consensus)", "la métrique principale du captioning"],
  ["SPICE", "graphe de scène : objets, attributs, relations", "correction sémantique"],
], { x: 0.6, y: 1.6, w: 12.1, colW: [2.2, 5.4, 4.5], fontFace: "Calibri",
  fontSize: 13.5, color: INK, border: { color: "C9D9E4", pt: 0.75 },
  fill: { color: WHT }, rowH: 0.55, valign: "middle", margin: 0.06 });
bullets(s, [
  { t: "Le projet calcule BLEU-1..4 nativement (evaluate.py) ; METEOR/CIDEr/SPICE via pycocoevalcap." },
  { t: "Règle d'or : toujours lire de vraies captions en plus des chiffres — c'est le rôle de error_analysis.py.", b: true },
], { x: 0.6, y: 5.15, w: 12.1, h: 1.6 });
note(s, "BLEU compte les mots communs ; CIDEr pondère par l'information ; SPICE compare le sens.");

// ============ S11 — Comparaison ============
s = pres.addSlide({ masterName: "LIGHT" });
title(s, "Trois décodeurs face à face");
s.addTable([
  [{ text: "Modèle", options: { bold: true, color: WHT, fill: { color: P } } },
   { text: "Paramètres", options: { bold: true, color: WHT, fill: { color: P } } },
   { text: "Entraînement", options: { bold: true, color: WHT, fill: { color: P } } },
   { text: "Force", options: { bold: true, color: WHT, fill: { color: P } } },
   { text: "BLEU-4 attendu*", options: { bold: true, color: WHT, fill: { color: P } } }],
  ["CNN + LSTM", "3,7 M", "séquentiel", "simple, rapide à coder", "0,17 – 0,21"],
  ["+ Attention", "7,1 M", "séquentiel (plus lent)", "interprétable (heatmaps)", "0,21 – 0,25"],
  ["Transformer", "13,7 M", "parallèle (rapide)", "meilleure qualité", "0,22 – 0,27"],
], { x: 0.6, y: 1.7, w: 12.1, colW: [2.5, 1.9, 2.6, 2.9, 2.2],
  fontFace: "Calibri", fontSize: 14, color: INK,
  border: { color: "C9D9E4", pt: 0.75 }, rowH: 0.62, valign: "middle",
  margin: 0.06 });
s.addText("* fourchettes typiques sur Flickr8k — le tableau réel se génère avec compare_models.py après entraînement.",
  { x: 0.6, y: 4.6, w: 12.1, h: 0.4, fontFace: "Calibri", fontSize: 12,
    italic: true, color: MUT, margin: 0 });
bullets(s, [
  { t: "Le plus grand saut de qualité de l'histoire du domaine : l'ajout de l'attention." },
  { t: "Le Transformer gagne surtout avec plus de données — d'où le pré-entraînement massif (BLIP, GPT-4V)." },
], { x: 0.6, y: 5.3, w: 12.1, h: 1.4 });
note(s, "Chiffres de paramètres mesurés sur le projet ; BLEU à remplir après entraînement Flickr8k.");

// ============ S12 — Analyse d'erreurs ============
s = pres.addSlide({ masterName: "LIGHT" });
title(s, "Analyse d'erreurs automatisée");
const cats = [
  ["Répétition", "« a dog dog on the … » — un bigramme revient en boucle"],
  ["Trop court", "moins de 4 mots : le modèle « abandonne »"],
  ["<unk>", "mots hors vocabulaire — augmenter les données ou baisser le seuil"],
  ["Générique", "la même caption pour des images différentes"],
];
cats.forEach((c, i) => {
  const x = 0.6 + (i % 2) * 6.25, y = 1.6 + Math.floor(i / 2) * 1.75;
  s.addShape("roundRect", { x, y, w: 5.85, h: 1.45, rectRadius: 0.08,
    fill: { color: ICE }, line: { color: "C9D9E4", width: 0.75 } });
  s.addText(c[0], { x: x + 0.3, y: y + 0.18, w: 5.3, h: 0.45,
    fontFace: "Calibri", fontSize: 16, bold: true, color: P, margin: 0 });
  s.addText(c[1], { x: x + 0.3, y: y + 0.68, w: 5.3, h: 0.65,
    fontFace: "Calibri", fontSize: 12.5, color: INK, margin: 0 });
});
s.addShape("roundRect", { x: 0.6, y: 5.25, w: 12.1, h: 1.45, rectRadius: 0.08,
  fill: { color: M }, line: { type: "none" } });
s.addText([
  { text: "Étude de cas réelle : ", options: { bold: true, color: "02C39A" } },
  { text: "sur le dataset de test factice, error_analysis.py a diagnostiqué 100 % de captions « génériques » — le modèle disait la même phrase pour toutes les images. Un chiffre qui raconte immédiatement le problème.",
    options: { color: WHT } },
], { x: 0.95, y: 5.45, w: 11.4, h: 1.05, fontFace: "Calibri", fontSize: 14,
  margin: 0 });
note(s, "error_analysis.py score chaque caption, détecte 4 patterns d'échec et sort les 10 pires / meilleures.");

// ============ S13 — Démo ============
s = pres.addSlide({ masterName: "LIGHT" });
title(s, "Démo : l'app Streamlit");
bullets(s, [
  { t: "Upload d'une photo → caption générée en direct." },
  { t: "Sélecteur de modèle (lstm / attention / transformer)." },
  { t: "Beam search réglable, mode greedy." },
  { t: "« Voir où le modèle regarde » : heatmap d'attention par mot." },
  { t: "Alternatives incluses dans le guide : API FastAPI, Hugging Face Spaces.", plain: true, s: 13, c: MUT },
], { x: 0.6, y: 1.6, w: 6.4, h: 4.2 });
s.addShape("roundRect", { x: 7.6, y: 1.8, w: 5.0, h: 2.0, rectRadius: 0.08,
  fill: { color: "0E1526" }, line: { type: "none" } });
s.addText([
  { text: "$ ", options: { color: "02C39A", bold: true } },
  { text: "cd image-captioning\n", options: { color: WHT } },
  { text: "$ ", options: { color: "02C39A", bold: true } },
  { text: "streamlit run app.py", options: { color: WHT } },
], { x: 7.95, y: 2.15, w: 4.4, h: 1.3, fontFace: "Courier New",
  fontSize: 15, margin: 0 });
s.addText("Une démo vivante vaut dix slides : générer une caption sur une photo du jury.",
  { x: 7.6, y: 4.3, w: 5.0, h: 1.0, fontFace: "Calibri", fontSize: 14,
    italic: true, color: MUT, margin: 0 });
note(s, "Lancer la démo en live : uploader une photo prise dans la salle.");

// ============ S14 — Leçons & perspectives (sombre) ============
s = pres.addSlide({ masterName: "DARK" });
title(s, "Leçons apprises & perspectives", true);
s.addText("Leçons de debugging", { x: 0.6, y: 1.5, w: 5.9, h: 0.5,
  fontFace: "Calibri", fontSize: 18, bold: true, color: "02C39A", margin: 0 });
bullets(s, [
  { t: "Un contexte 2048-d écrase un embedding 256-d : équilibrer les échelles.", c: WHT, s: 14 },
  { t: "LayerNorm sur la mémoire image — sans elle, le Transformer plafonne.", c: WHT, s: 14 },
  { t: "Le test « overfitter un seul batch » sépare bug et lenteur.", c: WHT, s: 14 },
  { t: "Checkpoints atomiques : une interruption ne corrompt plus rien.", c: WHT, s: 14 },
], { x: 0.6, y: 2.1, w: 5.9, h: 3.6 });
s.addText("Extensions possibles", { x: 6.9, y: 1.5, w: 5.8, h: 0.5,
  fontFace: "Calibri", fontSize: 18, bold: true, color: "02C39A", margin: 0 });
bullets(s, [
  { t: "Fine-tuner l'encodeur (phase 2 du transfer learning).", c: WHT, s: 14 },
  { t: "SCST : optimiser CIDEr par renforcement.", c: WHT, s: 14 },
  { t: "Comparer avec BLIP-2 fine-tuné : classique vs moderne.", c: WHT, s: 14 },
  { t: "Captioning multilingue (français, arabe).", c: WHT, s: 14 },
], { x: 6.9, y: 2.1, w: 5.8, h: 3.6 });
s.addText("Merci — place à la démo.", { x: 0.6, y: 6.3, w: 12.1, h: 0.6,
  align: "center", fontFace: "Cambria", fontSize: 22, bold: true,
  color: WHT, margin: 0 });
note(s, "Terminer sur les bugs réels trouvés : c'est ce qui montre la maîtrise. Puis démo live.");

pres.writeFile({ fileName: __dirname + "/presentation.pptx" })
  .then(() => console.log("presentation.pptx écrit"));
