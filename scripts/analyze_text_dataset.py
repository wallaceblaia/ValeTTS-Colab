#!/usr/bin/env python3
"""
Analisador de Dataset de Texto para TTS
======================================

Analisa a qualidade de um arquivo de texto para treinamento TTS:
- Verifica duplicatas
- Analisa comprimento das frases
- Avalia diversidade fonética
- Identifica problemas potenciais
- Fornece estatísticas detalhadas
"""

import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


def clean_text(text: str) -> str:
    """Limpa e normaliza o texto."""
    # Remove espaços extras
    text = re.sub(r"\s+", " ", text.strip())
    # Remove caracteres especiais problemáticos
    text = re.sub(r"[^\w\s\.\,\!\?\;\:\'\-]", "", text)
    return text


def count_phonetic_diversity(sentences: List[str]) -> Dict[str, int]:
    """Conta a diversidade fonética aproximada baseada em caracteres."""
    phonetic_coverage = defaultdict(int)

    for sentence in sentences:
        # Converte para minúsculas e remove pontuação
        clean_sentence = re.sub(r"[^a-zA-Z\s]", "", sentence.lower())

        # Conta bigramas e trigramas de letras (aproximação fonética)
        words = clean_sentence.split()
        for word in words:
            # Letras individuais
            for char in word:
                phonetic_coverage[f"char_{char}"] += 1

            # Bigramas
            for i in range(len(word) - 1):
                bigram = word[i : i + 2]
                phonetic_coverage[f"bigram_{bigram}"] += 1

            # Trigramas
            for i in range(len(word) - 2):
                trigram = word[i : i + 3]
                phonetic_coverage[f"trigram_{trigram}"] += 1

    return dict(phonetic_coverage)


def analyze_text_quality(sentences: List[str]) -> Dict:
    """Analisa a qualidade geral do texto para TTS."""
    analysis = {
        "total_sentences": len(sentences),
        "duplicates": {},
        "length_stats": {},
        "phonetic_diversity": {},
        "quality_issues": [],
        "character_distribution": {},
        "sentence_types": {},
        "recommendations": [],
    }

    # 1. Verificar duplicatas
    sentence_counts = Counter(sentences)
    duplicates = {sent: count for sent, count in sentence_counts.items() if count > 1}
    analysis["duplicates"] = {
        "count": len(duplicates),
        "examples": list(duplicates.items())[:10],  # Primeiros 10 exemplos
        "total_duplicate_sentences": sum(duplicates.values()) - len(duplicates),
    }

    # 2. Estatísticas de comprimento
    lengths = [len(sentence) for sentence in sentences]
    analysis["length_stats"] = {
        "min_length": min(lengths),
        "max_length": max(lengths),
        "mean_length": statistics.mean(lengths),
        "median_length": statistics.median(lengths),
        "std_dev": statistics.stdev(lengths) if len(lengths) > 1 else 0,
        "length_distribution": dict(Counter(lengths)),
    }

    # 3. Filtrar por critério TTS (5-300 caracteres OU máximo 40 palavras)
    def is_ideal_for_tts(sentence):
        char_count = len(sentence)
        word_count = len(sentence.split())
        return 5 <= char_count <= 300 and word_count <= 40

    ideal_sentences = [s for s in sentences if is_ideal_for_tts(s)]
    too_short = [s for s in sentences if len(s) < 5]
    too_long = [s for s in sentences if len(s) > 300 or len(s.split()) > 40]

    analysis["tts_suitability"] = {
        "ideal_count": len(ideal_sentences),
        "ideal_percentage": len(ideal_sentences) / len(sentences) * 100,
        "too_short_count": len(too_short),
        "too_long_count": len(too_long),
        "too_short_examples": too_short[:5],
        "too_long_examples": too_long[:5],
    }

    # 4. Diversidade fonética
    phonetic_coverage = count_phonetic_diversity(sentences)
    unique_chars = len([k for k in phonetic_coverage.keys() if k.startswith("char_")])
    unique_bigrams = len(
        [k for k in phonetic_coverage.keys() if k.startswith("bigram_")]
    )

    analysis["phonetic_diversity"] = {
        "unique_characters": unique_chars,
        "unique_bigrams": unique_bigrams,
        "coverage_score": min(
            100, (unique_chars / 26) * 100
        ),  # Baseado no alfabeto inglês
    }

    # 5. Distribuição de caracteres
    char_counts = defaultdict(int)
    for sentence in sentences:
        for char in sentence.lower():
            if char.isalpha():
                char_counts[char] += 1

    analysis["character_distribution"] = dict(char_counts)

    # 6. Tipos de sentenças
    questions = sum(1 for s in sentences if "?" in s)
    exclamations = sum(1 for s in sentences if "!" in s)
    statements = len(sentences) - questions - exclamations

    analysis["sentence_types"] = {
        "statements": statements,
        "questions": questions,
        "exclamations": exclamations,
        "statements_pct": statements / len(sentences) * 100,
        "questions_pct": questions / len(sentences) * 100,
        "exclamations_pct": exclamations / len(sentences) * 100,
    }

    # 7. Identificar problemas de qualidade
    issues = []

    if analysis["duplicates"]["count"] > len(sentences) * 0.1:
        issues.append(
            f"Alto número de duplicatas: {analysis['duplicates']['count']} ({analysis['duplicates']['count']/len(sentences)*100:.1f}%)"
        )

    if analysis["tts_suitability"]["ideal_percentage"] < 80:
        issues.append(
            f"Apenas {analysis['tts_suitability']['ideal_percentage']:.1f}% das frases são ideais para TTS (5-40 caracteres)"
        )

    if analysis["phonetic_diversity"]["coverage_score"] < 80:
        issues.append(
            f"Cobertura fonética baixa: {analysis['phonetic_diversity']['coverage_score']:.1f}%"
        )

    if analysis["sentence_types"]["questions_pct"] < 5:
        issues.append("Poucas perguntas - falta diversidade de entonação")

    if analysis["sentence_types"]["exclamations_pct"] < 2:
        issues.append("Poucas exclamações - falta expressividade")

    analysis["quality_issues"] = issues

    # 8. Recomendações
    recommendations = []

    if analysis["duplicates"]["count"] > 0:
        recommendations.append("Remover duplicatas para melhorar diversidade")

    if analysis["tts_suitability"]["too_long_count"] > 0:
        recommendations.append(
            f"Dividir {analysis['tts_suitability']['too_long_count']} frases longas em menores"
        )

    if analysis["tts_suitability"]["too_short_count"] > 0:
        recommendations.append(
            f"Expandir ou remover {analysis['tts_suitability']['too_short_count']} frases muito curtas"
        )

    if analysis["phonetic_diversity"]["coverage_score"] < 90:
        recommendations.append("Adicionar mais frases com diversidade fonética")

    if analysis["sentence_types"]["questions_pct"] < 10:
        recommendations.append("Adicionar mais perguntas para variedade de entonação")

    analysis["recommendations"] = recommendations

    return analysis


def print_analysis_report(analysis: Dict, filename: str):
    """Imprime relatório detalhado da análise."""
    print(f"\n{'='*60}")
    print(f"ANÁLISE DE QUALIDADE TTS - {filename}")
    print(f"{'='*60}")

    # Estatísticas gerais
    print(f"\n📊 ESTATÍSTICAS GERAIS:")
    print(f"   Total de frases: {analysis['total_sentences']:,}")
    print(
        f"   Frases únicas: {analysis['total_sentences'] - analysis['duplicates']['total_duplicate_sentences']:,}"
    )
    print(f"   Duplicatas encontradas: {analysis['duplicates']['count']:,}")

    # Adequação para TTS
    print(f"\n🎯 ADEQUAÇÃO PARA TTS (5-40 caracteres):")
    print(
        f"   Frases ideais: {analysis['tts_suitability']['ideal_count']:,} ({analysis['tts_suitability']['ideal_percentage']:.1f}%)"
    )
    print(f"   Muito curtas: {analysis['tts_suitability']['too_short_count']:,}")
    print(f"   Muito longas: {analysis['tts_suitability']['too_long_count']:,}")

    # Comprimento das frases
    print(f"\n📏 COMPRIMENTO DAS FRASES:")
    print(f"   Mínimo: {analysis['length_stats']['min_length']} caracteres")
    print(f"   Máximo: {analysis['length_stats']['max_length']} caracteres")
    print(f"   Média: {analysis['length_stats']['mean_length']:.1f} caracteres")
    print(f"   Mediana: {analysis['length_stats']['median_length']:.1f} caracteres")

    # Diversidade fonética
    print(f"\n🔤 DIVERSIDADE FONÉTICA:")
    print(
        f"   Caracteres únicos: {analysis['phonetic_diversity']['unique_characters']}/26"
    )
    print(f"   Bigramas únicos: {analysis['phonetic_diversity']['unique_bigrams']}")
    print(
        f"   Score de cobertura: {analysis['phonetic_diversity']['coverage_score']:.1f}%"
    )

    # Tipos de sentenças
    print(f"\n📝 TIPOS DE SENTENÇAS:")
    print(
        f"   Declarações: {analysis['sentence_types']['statements']:,} ({analysis['sentence_types']['statements_pct']:.1f}%)"
    )
    print(
        f"   Perguntas: {analysis['sentence_types']['questions']:,} ({analysis['sentence_types']['questions_pct']:.1f}%)"
    )
    print(
        f"   Exclamações: {analysis['sentence_types']['exclamations']:,} ({analysis['sentence_types']['exclamations_pct']:.1f}%)"
    )

    # Problemas identificados
    if analysis["quality_issues"]:
        print(f"\n⚠️  PROBLEMAS IDENTIFICADOS:")
        for i, issue in enumerate(analysis["quality_issues"], 1):
            print(f"   {i}. {issue}")
    else:
        print(f"\n✅ QUALIDADE: Nenhum problema crítico encontrado!")

    # Duplicatas (exemplos)
    if analysis["duplicates"]["count"] > 0:
        print(f"\n🔄 EXEMPLOS DE DUPLICATAS:")
        for sentence, count in analysis["duplicates"]["examples"][:5]:
            print(f"   '{sentence}' (aparece {count}x)")
        if len(analysis["duplicates"]["examples"]) > 5:
            print(
                f"   ... e mais {len(analysis['duplicates']['examples']) - 5} duplicatas"
            )

    # Frases muito longas (exemplos)
    if analysis["tts_suitability"]["too_long_count"] > 0:
        print(f"\n📏 EXEMPLOS DE FRASES MUITO LONGAS:")
        for sentence in analysis["tts_suitability"]["too_long_examples"]:
            print(f"   '{sentence}' ({len(sentence)} chars)")

    # Recomendações
    if analysis["recommendations"]:
        print(f"\n💡 RECOMENDAÇÕES:")
        for i, rec in enumerate(analysis["recommendations"], 1):
            print(f"   {i}. {rec}")

    # Score geral
    score = 100
    if analysis["duplicates"]["count"] > 0:
        score -= min(
            20, analysis["duplicates"]["count"] / analysis["total_sentences"] * 100
        )
    if analysis["tts_suitability"]["ideal_percentage"] < 80:
        score -= 80 - analysis["tts_suitability"]["ideal_percentage"]
    if analysis["phonetic_diversity"]["coverage_score"] < 80:
        score -= (80 - analysis["phonetic_diversity"]["coverage_score"]) / 2

    score = max(0, score)

    print(f"\n🏆 SCORE GERAL DE QUALIDADE: {score:.1f}/100")

    if score >= 90:
        print("   Excelente! Dataset pronto para TTS")
    elif score >= 75:
        print("   Bom! Algumas melhorias recomendadas")
    elif score >= 60:
        print("   Médio! Necessita melhorias significativas")
    else:
        print("   Baixo! Requer reestruturação substancial")

    print(f"\n{'='*60}")


def main():
    """Função principal."""
    if len(sys.argv) != 2:
        print("Uso: python analyze_text_dataset.py <arquivo_texto>")
        sys.exit(1)

    filename = sys.argv[1]
    filepath = Path(filename)

    if not filepath.exists():
        print(f"Erro: Arquivo '{filename}' não encontrado")
        sys.exit(1)

    try:
        # Ler arquivo
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Limpar e filtrar linhas
        sentences = []
        for line in lines:
            cleaned = clean_text(line.strip())
            if cleaned:  # Ignorar linhas vazias
                sentences.append(cleaned)

        if not sentences:
            print("Erro: Arquivo não contém frases válidas")
            sys.exit(1)

        # Analisar
        print("Analisando dataset de texto...")
        analysis = analyze_text_quality(sentences)

        # Mostrar relatório
        print_analysis_report(analysis, filename)

    except Exception as e:
        print(f"Erro ao processar arquivo: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
