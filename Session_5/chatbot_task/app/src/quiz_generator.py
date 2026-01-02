"""
Quiz-Generator Modul
Generiert Multiple-Choice Fragen aus ausgewählten PDFs basierend auf Chat-Kontext.
"""

import logging
from typing import List, Dict, Any
import json
import re

logger = logging.getLogger(__name__)


class QuizGenerator:
    """Generiert Quiz-Fragen aus PDF-Chunks mit LLM."""
    
    def __init__(self, chatbot):
        """
        Initialisiert den Quiz-Generator.
        
        Args:
            chatbot: CustomChatBot Instanz für LLM-Zugriff
        """
        self.chatbot = chatbot
        self.llm = chatbot.llm
    
    def generate_quiz(self, selected_pdfs: List[str], chat_context: str = "", num_questions: int = None) -> Dict[str, Any]:
        """
        Generiert ein Quiz basierend auf selected PDFs und Chat-Kontext.
        
        Args:
            selected_pdfs: Liste der ausgewählten PDF-Namen
            chat_context: Optionaler Chat-Kontext/Thema
            num_questions: Anzahl der zu generierenden Fragen (3-10, auto wenn None)
            
        Returns:
            dict mit quiz_data: Fragen, Optionen, richtige Antworten, etc.
        """
        
        if not selected_pdfs:
            logger.error("Keine PDFs ausgewählt für Quiz-Generierung")
            return None
        
        logger.info(f"Generiere Quiz für PDFs: {selected_pdfs}")
        
        try:
            # Abrufen der relevanten Chunks aus den ausgewählten PDFs
            relevant_chunks = self._get_relevant_chunks(selected_pdfs)
            
            if not relevant_chunks:
                logger.warning("Keine Chunks für Quiz-Generierung gefunden")
                return None
            
            # Bestimme Anzahl der Fragen basierend auf Chunk-Menge
            if num_questions is None:
                num_questions = self._calculate_question_count(relevant_chunks)
            
            logger.info(f"Generiere {num_questions} Fragen aus {len(relevant_chunks)} Chunks")
            
            # Prompt für Quiz-Generierung
            quiz_prompt = self._build_quiz_prompt(
                relevant_chunks, 
                num_questions, 
                chat_context,
                selected_pdfs
            )
            
            # Generiere Fragen mit LLM
            logger.info("Rufe LLM auf um Fragen zu generieren...")
            response = self.llm.invoke(quiz_prompt).content
            
            # Parse die generierten Fragen
            quiz_data = self._parse_quiz_response(response, relevant_chunks)
            
            if not quiz_data or not quiz_data.get("questions"):
                logger.error("Fehler beim Parsen der Quiz-Antwort")
                return None
            
            # Füge Metadaten hinzu
            quiz_data["source_pdfs"] = selected_pdfs
            quiz_data["theme"] = chat_context or ", ".join(selected_pdfs)
            quiz_data["total_questions"] = len(quiz_data["questions"])
            
            logger.info(f"✓ Quiz erfolgreich generiert mit {quiz_data['total_questions']} Fragen")
            return quiz_data
            
        except Exception as e:
            logger.error(f"Fehler bei Quiz-Generierung: {e}", exc_info=True)
            return None
    
    def _get_relevant_chunks(self, selected_pdfs: List[str], chunk_count: int = 30) -> List[Dict[str, str]]:
        """
        Ruft relevante Chunks aus den ausgewählten PDFs ab.
        Versucht mehrere Suchanfragen um diverse Inhalte zu bekommen.
        
        Args:
            selected_pdfs: Liste der PDF-Namen
            chunk_count: Max. Anzahl der abzurufenden Chunks (30 max für LLM-Limits)
            
        Returns:
            Liste mit Chunk-Inhalten und Quellen
        """
        try:
            retriever = self.chatbot.get_filtered_retriever(selected_pdfs)
            
            # Mehrere Such-Queries um vielfältigere Inhalte zu bekommen
            # Aber begrenzt wegen LLM-Token-Limits
            queries = [
                "Wichtigste Konzepte und Definitionen",
                "Hauptthemen und Techniken",
                "Grundlagen und Erklärungen",
                "Schlüsselbegriffe und Methoden"
            ]
            
            relevant_chunks = {}  # Verwende Dict um Duplikate zu vermeiden
            
            for query in queries:
                try:
                    chunks = retriever.invoke(query)
                    
                    # Pro Query max 5-6 Chunks um LLM nicht zu überfordern
                    for chunk in chunks[:6]:
                        chunk_text = chunk.page_content
                        # Verwende Chunk-Text als Key um Duplikate zu vermeiden
                        if chunk_text not in relevant_chunks:
                            relevant_chunks[chunk_text] = {
                                "content": chunk_text,
                                "source": chunk.metadata.get("source_file", "Unknown")
                            }
                            
                            # Stoppe wenn wir genug haben (max 30)
                            if len(relevant_chunks) >= chunk_count:
                                break
                except Exception as e:
                    logger.warning(f"Fehler bei Query '{query}': {e}")
                    continue
                
                if len(relevant_chunks) >= chunk_count:
                    break
            
            result = list(relevant_chunks.values())
            logger.info(f"Abgerufen {len(result)} relevante Chunks aus {len(selected_pdfs)} PDFs")
            return result
            
        except Exception as e:
            logger.error(f"Fehler beim Abrufen von Chunks: {e}")
            return []
    
    def _calculate_question_count(self, chunks: List[Dict]) -> int:
        """
        Berechnet die Anzahl der Fragen basierend auf Chunk-Menge.
        
        Args:
            chunks: Liste der verfügbaren Chunks
            
        Returns:
            Fragen-Anzahl zwischen 5 und 10
        """
        chunk_count = len(chunks)
        
        if chunk_count < 10:
            return 5
        elif chunk_count < 20:
            return 7
        else:
            return 10
    
    def _build_quiz_prompt(self, chunks: List[Dict], num_questions: int, 
                           chat_context: str, selected_pdfs: List[str]) -> str:
        """
        Erstellt den Prompt für LLM zur Fragen-Generierung.
        
        Args:
            chunks: Relevante Inhalts-Chunks
            num_questions: Anzahl zu generierender Fragen
            chat_context: Chat-Thema/Kontext
            selected_pdfs: Quell-PDFs
            
        Returns:
            Formatted Prompt für LLM
        """
        
        chunk_text = "\n\n".join([f"{c['content'][:300]}" for c in chunks[:15]])
        
        prompt = f"""Erstelle exakt {num_questions} Multiple-Choice Fragen mit 4 kurzen Antworten.

QUELLE-TEXT (max 300 Zeichen pro Chunk):
{chunk_text}

ANLEITUNG:
- Jede Frage hat GENAU 4 Antwortoptionen
- Jede Antwort ist MAX 100 Zeichen lang
- Genau EINE Antwort ist richtig
- Nur valides JSON ausgeben

BEISPIEL:
[
  {{"question": "Was ist X?", "options": ["Option 1", "Option 2", "Option 3", "Option 4"], "correct_index": 0, "difficulty": "easy", "explanation": "Erklaerung"}},
  {{"question": "Wie funktioniert Y?", "options": ["Antwort A", "Antwort B", "Antwort C", "Antwort D"], "correct_index": 2, "difficulty": "medium", "explanation": "Weil..."}}
]

Ausgabe: NUR das JSON-Array, Punkt."""
        
        return prompt
    
    def _parse_quiz_response(self, response: str, chunks: List[Dict]) -> Dict[str, Any]:
        """
        Parsed die LLM-Antwort in Quiz-Datenstruktur.
        
        Args:
            response: LLM-Response mit JSON
            chunks: Originale Chunks für Source-Zuordnung
            
        Returns:
            Strukturierte Quiz-Daten
        """
        
        try:
            logger.info(f"Parsing LLM Response ({len(response)} Zeichen)")
            
            # Bereinige die Response
            cleaned_response = response.strip()
            
            # Entferne Markdown Code-Blöcke falls vorhanden
            cleaned_response = re.sub(r'```json\s*', '', cleaned_response)
            cleaned_response = re.sub(r'```\s*', '', cleaned_response)
            
            # Entferne ungültige Escape-Sequenzen wie {\[ oder \]
            cleaned_response = cleaned_response.replace('\\[', '[').replace('\\]', ']')
            cleaned_response = cleaned_response.replace('{\\[', '[').replace('\\]}', ']')
            
            # Entferne führende Zeichen vor dem JSON-Array
            # Finde das erste '[' und schneide alles davor ab
            first_bracket = cleaned_response.find('[')
            if first_bracket > 0:
                cleaned_response = cleaned_response[first_bracket:]
            
            # Extrahiere JSON-Array aus Response
            json_match = re.search(r'\[\s*\{.*\}\s*\]', cleaned_response, re.DOTALL)
            if not json_match:
                logger.error(f"Keine JSON in Response gefunden. Response: {cleaned_response[:500]}")
                # Fallback: Versuche einzelne Fragen zu extrahieren
                questions_raw = self._extract_questions_fallback(response)
                if questions_raw:
                    logger.info(f"Fallback erfolgreich: {len(questions_raw)} Fragen extrahiert")
                else:
                    return None
            else:
                json_str = json_match.group(0)
                
                # Versuche JSON zu reparieren (häufige Fehler)
                json_str = re.sub(r',\s*]', ']', json_str)  # Trailing comma
                json_str = re.sub(r',\s*}', '}', json_str)  # Trailing comma in object
                json_str = json_str.replace('\n', ' ')  # Newlines entfernen
                
                try:
                    questions_raw = json.loads(json_str)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON Parse-Fehler: {e}")
                    logger.error(f"JSON String: {json_str[:500]}")
                    
                    # Fallback: Versuche einzelne Fragen zu extrahieren
                    questions_raw = self._extract_questions_fallback(response)
                    if not questions_raw:
                        return None
            
            # Verarbeite Fragen
            questions = []
            for idx, q in enumerate(questions_raw):
                question_obj = {
                    "id": idx + 1,
                    "question": q.get("question", ""),
                    "options": q.get("options", []),
                    "correct_index": int(q.get("correct_index", 0)),
                    "difficulty": q.get("difficulty", "medium"),
                    "explanation": q.get("explanation", ""),
                    "source_chunk": self._find_best_chunk(q.get("question", ""), chunks),
                    "user_answer": None,
                    "is_correct": None,
                    "attempts": 0
                }
                questions.append(question_obj)
            
            return {
                "questions": questions,
                "current_question_index": 0,
                "failed_questions": []
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON Parse-Fehler: {e}")
            return None
        except Exception as e:
            logger.error(f"Fehler beim Parsen der Quiz-Antwort: {e}", exc_info=True)
            return None
    
    def _find_best_chunk(self, question: str, chunks: List[Dict], top_k: int = 1) -> str:
        """
        Findet den besten Chunk als Quelle für eine Frage.
        
        Args:
            question: Die generierte Frage
            chunks: Verfügbare Chunks
            top_k: Anzahl der Top-Chunks
            
        Returns:
            Bester Chunk-Text oder leer
        """
        
        if not chunks:
            return ""
        
        try:
            # Einfache Text-Ähnlichkeit: Zähle gemeinsame Wörter
            question_words = set(question.lower().split())
            
            best_chunk = chunks[0]["content"]
            best_score = 0
            
            for chunk in chunks:
                chunk_words = set(chunk["content"].lower().split())
                similarity = len(question_words & chunk_words)
                
                if similarity > best_score:
                    best_score = similarity
                    best_chunk = chunk["content"][:200]  # Limit auf 200 Zeichen
            
            return best_chunk
            
        except Exception as e:
            logger.error(f"Fehler beim Finden des besten Chunks: {e}")
            return chunks[0]["content"][:200] if chunks else ""

    def _extract_questions_fallback(self, response: str) -> List[Dict]:
        """
        Fallback-Methode um Fragen aus verschiedenen Formaten zu extrahieren.
        Unterstützt: JSON, Text mit "Frage X", "1.", "A)", "B)", "C)", "D)"
        
        Args:
            response: LLM Response Text
            
        Returns:
            Liste von Fragen-Dicts oder leere Liste
        """
        try:
            questions = []
            
            # Versuche zuerst strukturiertes Text-Format mit Nummern (1., 2., 3., etc.)
            # Muster: "1. Was ist..." bis zur nächsten Nummer oder Ende
            numbered_pattern = r'^\s*\d+[\.\)]\s+(.+?)(?=^\s*\d+[\.\)]|$)'
            
            numbered_blocks = list(re.finditer(numbered_pattern, response, re.MULTILINE | re.DOTALL))
            
            if numbered_blocks:
                for num_match in numbered_blocks:
                    block = num_match.group(0)
                    
                    # Extrahiere Frage (erste Zeile)
                    lines = block.strip().split('\n')
                    question_text = lines[0].strip()
                    
                    # Entferne Fragen-Nummer von der Frage
                    question_text = re.sub(r'^\s*\d+[\.\)]\s*', '', question_text)
                    
                    if not question_text or len(question_text) < 5:
                        continue
                    
                    # Extrahiere Optionen A), B), C), D)
                    option_pattern = r'[A-D]\)\s*(.+?)(?=[A-D]\)|Antwort|Answer|Lösung|$)'
                    option_matches = list(re.finditer(option_pattern, block, re.DOTALL))
                    
                    options = []
                    for opt_match in option_matches:
                        opt_text = opt_match.group(1).strip()
                        opt_text = ' '.join(opt_text.split())[:100]
                        if opt_text:
                            options.append(opt_text)
                    
                    if len(options) >= 4:
                        # Versuche richtige Antwort zu finden
                        answer_pattern = r'(?:Antwort|Answer|Lösung|Correct)\s*[:\.]?\s*([A-D])\)'
                        answer_match = re.search(answer_pattern, block, re.IGNORECASE)
                        correct_idx = 0
                        
                        if answer_match:
                            answer_letter = answer_match.group(1).upper()
                            correct_idx = ord(answer_letter) - ord('A')
                        
                        questions.append({
                            "question": question_text[:150],
                            "options": options[:4],
                            "correct_index": min(correct_idx, 3),
                            "difficulty": "medium",
                            "explanation": ""
                        })
                
                if questions:
                    logger.info(f"Fallback Nummern-Format: {len(questions)} Fragen extrahiert")
                    return questions
            
            # Versuche dann strukturiertes Text-Format mit "Frage X", "Question X"
            # Suche nach Frage-Mustern: "Frage 1:", "Frage 2:", etc
            frage_pattern = r'(?:Frage|Question)\s*\d+\s*[:\.]?\s*\n?(.+?)(?=(?:Frage|Question)\s*\d+|$)'
            
            frage_blocks = list(re.finditer(frage_pattern, response, re.IGNORECASE | re.DOTALL))
            
            if frage_blocks:
                for frage_match in frage_blocks:
                    block = frage_match.group(0)
                    
                    # Extrahiere Frage (erste Zeile die kein A), B), etc ist)
                    lines = block.strip().split('\n')
                    question_text = None
                    
                    for line in lines:
                        line_clean = line.strip()
                        if line_clean and not line_clean[0] in ('A', 'B', 'C', 'D', 'a', 'b', 'c', 'd'):
                            question_text = line_clean
                            break
                    
                    if not question_text:
                        continue
                    
                    # Extrahiere Optionen A), B), C), D)
                    option_pattern = r'[A-D]\)\s*(.+?)(?=[A-D]\)|Antwort|Answer|$)'
                    option_matches = list(re.finditer(option_pattern, block, re.DOTALL))
                    
                    options = []
                    for opt_match in option_matches:
                        opt_text = opt_match.group(1).strip()
                        opt_text = ' '.join(opt_text.split())[:100]
                        options.append(opt_text)
                    
                    if len(options) >= 4:
                        # Versuche richtige Antwort zu finden
                        answer_pattern = r'(?:Antwort|Answer|Correct)\s*[:\.]?\s*([A-D])\)'
                        answer_match = re.search(answer_pattern, block, re.IGNORECASE)
                        correct_idx = 0
                        
                        if answer_match:
                            answer_letter = answer_match.group(1).upper()
                            correct_idx = ord(answer_letter) - ord('A')
                        
                        questions.append({
                            "question": question_text[:150],
                            "options": options[:4],
                            "correct_index": min(correct_idx, 3),
                            "difficulty": "medium",
                            "explanation": ""
                        })
                
                if questions:
                    logger.info(f"Fallback Text-Format: {len(questions)} Fragen extrahiert")
                    return questions
            
            # Fallback zu JSON wenn Text-Format nicht gefunden
            if '"question"' in response:
                logger.info("Versuche JSON-Format zu parsen...")
                
                # Finde alle question-strings mit besserer Regex
                question_pattern = r'"question"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"'
                question_matches = list(re.finditer(question_pattern, response))
                
                logger.info(f"Gefundene Fragen im JSON: {len(question_matches)}")
                
                for i, qmatch in enumerate(question_matches):
                    question_text = qmatch.group(1).strip()
                    
                    if not question_text or len(question_text) < 3:
                        continue
                    
                    # Finde den Block nach dieser Frage bis zur nächsten oder zum Ende
                    start_pos = qmatch.end()
                    
                    # Finde die nächste "question" oder nutze das Ende
                    next_question = re.search(r'"question"\s*:', response[start_pos:])
                    if next_question:
                        end_pos = start_pos + next_question.start()
                    else:
                        end_pos = len(response)
                    
                    block = response[qmatch.start():end_pos]
                    
                    # Finde options array
                    opt_match = re.search(r'"options"\s*:\s*\[([^\]]*)\]', block, re.DOTALL)
                    if not opt_match:
                        continue
                    
                    options_str = opt_match.group(1)
                    
                    # Extrahiere alle Strings in den options
                    options = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', options_str)
                    
                    if len(options) < 4:
                        # Wenn nicht genug Optionen, skip diese Frage
                        logger.info(f"Überspringe Frage {i+1}: nur {len(options)} Optionen gefunden")
                        continue
                    
                    # Finde correct_index
                    c_match = re.search(r'"correct_index"\s*:\s*(\d+)', block)
                    correct_idx = int(c_match.group(1)) if c_match else 0
                    correct_idx = min(correct_idx, 3)
                    
                    # Finde difficulty
                    diff_match = re.search(r'"difficulty"\s*:\s*"([^"]*)"', block)
                    difficulty = diff_match.group(1) if diff_match else "medium"
                    
                    # Finde explanation
                    expl_match = re.search(r'"explanation"\s*:\s*"([^"]*)"', block)
                    explanation = expl_match.group(1) if expl_match else ""
                    
                    questions.append({
                        "question": question_text.strip(),
                        "options": [opt.strip() for opt in options[:4]],
                        "correct_index": correct_idx,
                        "difficulty": difficulty,
                        "explanation": explanation
                    })
                
                if questions:
                    logger.info(f"Fallback JSON: {len(questions)} Fragen extrahiert")
                    return questions
            
            return []
            
        except Exception as e:
            logger.error(f"Fallback-Extraktion fehlgeschlagen: {e}")
            return []

