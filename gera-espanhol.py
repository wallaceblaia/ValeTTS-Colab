#!/usr/bin/env python3
"""
gera-espanhol.py
------------------------------------------------------------
Gera 20 000 sentenças da Bíblia em espanhol (Reina Valera),
com 5–25 palavras, iniciando com letra maiúscula e finalizando
em ponto, exclamação ou interrogação.

Suporta tokenização com spaCy, NLTK ou Regex.

Requisitos:
    pip install spacy nltk tqdm
    python -m spacy download es_core_news_sm
"""

import re
import random
import pathlib
import urllib.request
import urllib.error
from typing import Iterable, List
from tqdm import tqdm

# ─── CONFIGURAÇÃO ────────────────────────────────────────────
BIBLE_URLS = [
    "https://raw.githubusercontent.com/bible-hub/Bibles/master/Spanish__Reina_Valera_(1909)__valera__LTR.txt",
]

TOKENIZER   = "spacy"        # "spacy" | "nltk" | "regex"
TARGET      = 18_000
MIN_W, MAX_W = 5, 25
DEST_FILE   = pathlib.Path("bible_es_20000.txt")
SEED        = 20250607
# ─────────────────────────────────────────────────────────────

# ─── Carrega tokenizador escolhido ───────────────────────────
def setup_tokenizer():
    if TOKENIZER == "spacy":
        import spacy
        try:
            nlp = spacy.load("es_core_news_sm")
        except OSError:
            raise RuntimeError(
                "Modelo spaCy 'es_core_news_sm' não encontrado.\n"
                "Instale com:  python -m spacy download es_core_news_sm"
            )
        return lambda txt: (s.text.strip() for s in nlp(txt).sents)

    if TOKENIZER == "nltk":
        import nltk
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")
        tokenizer = nltk.data.load("tokenizers/punkt/spanish.pickle")
        return lambda txt: tokenizer.tokenize(txt)

    SENT_RE = re.compile(r"[A-ZÁÉÍÓÚÑÜ][^.!?]*[.!?]")
    return lambda txt: (m.group() for m in SENT_RE.finditer(txt))


tokenize = setup_tokenizer()

# ─── Funções utilitárias ─────────────────────────────────────
def baixar(url: str) -> str:
    print(f"• Tentando {url}")
    try:
        with urllib.request.urlopen(url, timeout=45) as resp:
            print("  ✔️  Sucesso")
            return resp.read().decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"  ⚠️  Falhou: {e}")
        return ""

def limpar_versos_brutos(texto: str) -> str:
    linhas = texto.splitlines()
    versos_puros = []
    for linha in linhas:
        if "||" in linha:
            partes = linha.split("||")
            verso = partes[-1].strip()
        else:
            verso = linha.strip()
        if verso:
            versos_puros.append(verso)
    return "\n".join(versos_puros)

def filtrar_sentencas(sent_iter: Iterable[str]) -> List[str]:
    out = []
    for s in sent_iter:
        s = " ".join(s.split())
        if not s or s[0].islower() or s[-1] not in ".!?":
            continue
        n = len(s.split())
        if MIN_W <= n <= MAX_W:
            out.append(s)
    return out

# ─── Pipeline principal ─────────────────────────────────────
def main():
    random.seed(SEED)
    coletadas = set()

    for url in BIBLE_URLS:
        texto_raw = baixar(url)
        if not texto_raw:
            continue
        texto_limpado = limpar_versos_brutos(texto_raw)
        sent_iter = tokenize(texto_limpado)
        validas = filtrar_sentencas(sent_iter)
        coletadas.update(validas)
        print(f"Total elegíveis: {len(coletadas):,}")
        if len(coletadas) >= TARGET * 2:
            break

    if len(coletadas) < TARGET:
        raise RuntimeError("Menos versos que TARGET; afrouxe limites ou acrescente fontes.")

    selecionadas = random.sample(coletadas, TARGET)
    DEST_FILE.write_text("\n".join(selecionadas), encoding="utf-8")
    print(f"\n✔️  Gravado {TARGET} linhas em {DEST_FILE.resolve()}")

# ─── Execução ────────────────────────────────────────────────
if __name__ == "__main__":
    main()
