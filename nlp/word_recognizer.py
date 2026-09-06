"""
Assistive NLP Decoder & Word-Level Language Model.
Translates sequences of ambiguous character predictions into coherent words and phrases.
Supports beam search, Levenshtein distance error correction, and medical/assistive vocabulary.
"""

import heapq
from typing import Dict, List, Optional, Tuple
import numpy as np


ASSISTIVE_VOCABULARY = [
    # Medical & Emergency
    "HELP", "PAIN", "NURSE", "DOCTOR", "MEDICINE", "EMERGENCY", "HOSPITAL", "URGENT",
    "BREATHE", "FALL", "DIZZY", "TREMOR", "STIFF", "FREEZE", "HEADACHE", "MEDS",

    # Daily Care & Physical Needs
    "WATER", "FOOD", "HUNGRY", "THIRSTY", "BATHROOM", "REST", "SLEEP", "TIRED",
    "COLD", "HOT", "WALK", "SIT", "STAND", "LIE DOWN", "LIGHT", "BLANKET",

    # Social & Conversation
    "YES", "NO", "OK", "OKAY", "HI", "HELLO", "BYE", "GOODBYE", "PLEASE", "THANKS", "THANK YOU",
    "GOOD", "BAD", "WAIT", "COME", "GO", "STOP", "HAPPY", "LOVE", "FAMILY", "TALK", "AGAIN",
    "ME", "CALL", "BED", "SOUP", "TEA", "UP", "DOWN", "CAT", "DOG", "CAR", "BOOK", "PEN",
    "HURT", "HAND", "LEG", "HEAD", "EYE", "EAR", "FEET", "MORE", "LESS", "WARM", "NAME",

    # Common English Core
    "THE", "AND", "THAT", "HAVE", "FOR", "NOT", "WITH", "YOU", "THIS", "BUT",
    "FROM", "THEY", "WILL", "WOULD", "THERE", "WHAT", "WHEN", "MAKE", "CAN", "LIKE",
    "TIME", "JUST", "KNOW", "TAKE", "PEOPLE", "INTO", "YEAR", "YOUR", "GOOD", "SOME",
    "COULD", "THEM", "SEE", "OTHER", "THAN", "THEN", "NOW", "LOOK", "ONLY", "COME",
    "ITS", "OVER", "THINK", "ALSO", "BACK", "AFTER", "USE", "TWO", "HOW", "OUR",
    "WORK", "FIRST", "WELL", "WAY", "EVEN", "NEW", "WANT", "BECAUSE", "ANY", "THESE",
    "GIVE", "DAY", "MOST", "US"
]


class AssistiveNLPDecoder:
    """
    Performs word-level decoding, beam search, and vocabulary correction
    for assistive air-writing communication.
    """

    def __init__(self, vocabulary: Optional[List[str]] = None):
        self.vocab = set(w.upper() for w in (vocabulary or ASSISTIVE_VOCABULARY))
        self.vocab_list = sorted(list(self.vocab))

    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """Compute edit distance between two strings."""
        if len(s1) < len(s2):
            return AssistiveNLPDecoder.levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def correct_word(self, query: str, max_distance: int = 2) -> List[Tuple[str, float]]:
        """
        Find matching words in vocabulary with similarity score.
        """
        q = query.upper().strip()
        if not q:
            return []

        candidates = []
        for word in self.vocab_list:
            dist = self.levenshtein_distance(q, word)
            if dist <= max_distance:
                # Score based on length similarity and edit distance
                sim = 1.0 - (dist / max(len(q), len(word), 1))
                # Boost if query is prefix of word
                if word.startswith(q):
                    sim += 0.3
                candidates.append((word, float(sim)))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:5]

    def autocomplete_prefix(self, prefix: str, max_results: int = 5) -> List[str]:
        """
        Autocomplete words starting with prefix.
        """
        p = prefix.upper().strip()
        if not p:
            return self.vocab_list[:max_results]

        matches = [w for w in self.vocab_list if w.startswith(p)]
        # Sort by shortness
        matches.sort(key=lambda w: (len(w), w))
        return matches[:max_results]

    def beam_search_decode(
        self,
        char_posteriors: List[List[Tuple[str, float]]],
        beam_width: int = 4
    ) -> List[Tuple[str, float]]:
        """
        Beam search decoding across sequence of character candidate distributions.

        Args:
            char_posteriors: List of top candidates per stroke, e.g.
                             [[(H, 0.8), (N, 0.15)], [(E, 0.9), (F, 0.08)], ...]
        Returns:
            Top decoded word candidates with combined log probabilities.
        """
        if not char_posteriors:
            return []

        # Beam entry: (neg_log_prob, word_string)
        beam = [(0.0, "")]

        for candidates in char_posteriors:
            new_beam = []
            for score, prefix in beam:
                for char, prob in candidates:
                    p = max(prob, 1e-4)
                    new_score = score - np.log(p)
                    new_prefix = prefix + char
                    new_beam.append((new_score, new_prefix))

            # Keep top beam_width hypotheses
            new_beam.sort(key=lambda x: x[0])
            beam = new_beam[:beam_width]

        # Rerank beam by vocabulary presence & edit distance
        candidate_scores: Dict[str, float] = {}
        DIGIT_TO_LETTER = {'0': 'O', '1': 'I', '5': 'S', '8': 'B', '2': 'Z'}

        for neg_log_p, hypothesis in beam:
            prob = float(np.exp(-neg_log_p))

            # Direct vocabulary match
            if hypothesis in self.vocab:
                candidate_scores[hypothesis] = max(candidate_scores.get(hypothesis, 0.0), prob * 2.5)
            else:
                candidate_scores[hypothesis] = max(candidate_scores.get(hypothesis, 0.0), prob)

            # Check digit-to-letter phonetic/visual substitution (e.g. "0K" -> "OK", "F00D" -> "FOOD", "N0" -> "NO")
            alpha_hypothesis = "".join(DIGIT_TO_LETTER.get(c, c) for c in hypothesis)
            if alpha_hypothesis in self.vocab:
                candidate_scores[alpha_hypothesis] = max(candidate_scores.get(alpha_hypothesis, 0.0), prob * 2.2)

            # Check nearest vocabulary word
            corrections = self.correct_word(hypothesis, max_distance=1)
            for corr_word, sim in corrections[:2]:
                if corr_word in self.vocab:
                    candidate_scores[corr_word] = max(candidate_scores.get(corr_word, 0.0), prob * (1.1 + sim))

        reranked = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        # Normalize probabilities
        total_p = sum(p for _, p in reranked) + 1e-8
        return [(word, round(p / total_p, 4)) for word, p in reranked]
