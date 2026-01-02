"""
Quiz-Manager Modul
Verwaltet Quiz-Status, Antworten, Statistiken und Export/Import.
"""

import logging
from typing import Dict, List, Any
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class QuizManager:
    """Verwaltet Quiz-Zustand und Fortschritt."""
    
    def __init__(self):
        """Initialisiert den Quiz-Manager."""
        self.current_quiz: Dict[str, Any] = None
        self.user_answers: Dict[int, int] = {}  # {question_id: user_answer_index}
        self.failed_questions: List[int] = []
        self.start_time: datetime = None
        self.completion_time: datetime = None
    
    def start_quiz(self, quiz_data: Dict[str, Any]) -> bool:
        """
        Startet ein neues Quiz.
        
        Args:
            quiz_data: Quiz-Daten von QuizGenerator
            
        Returns:
            True wenn erfolgreich, False sonst
        """
        
        if not quiz_data or not quiz_data.get("questions"):
            logger.error("Ungültige Quiz-Daten")
            return False
        
        self.current_quiz = quiz_data.copy()
        self.user_answers = {}
        self.failed_questions = []
        self.start_time = datetime.now()
        self.completion_time = None
        
        logger.info(f"Quiz mit {len(self.current_quiz['questions'])} Fragen gestartet")
        return True
    
    def get_current_question(self) -> Dict[str, Any]:
        """
        Gibt die aktuelle Frage zurück.
        
        Returns:
            Aktuelle Quiz-Frage oder None
        """
        
        if not self.current_quiz:
            logger.warning("Kein aktives Quiz")
            return None
        
        current_idx = self.current_quiz.get("current_question_index", 0)
        questions = self.current_quiz.get("questions", [])
        
        if current_idx < len(questions):
            return questions[current_idx]
        
        return None
    
    def answer_question(self, question_id: int, answer_index: int) -> bool:
        """
        Speichert die Antwort des Nutzers für eine Frage.
        
        Args:
            question_id: ID der Frage (1-based)
            answer_index: Index der gewählten Antwort (0-3)
            
        Returns:
            True wenn Antwort korrekt
        """
        
        if not self.current_quiz:
            logger.error("Kein aktives Quiz zum Beantworten")
            return False
        
        questions = self.current_quiz.get("questions", [])
        question_idx = question_id - 1  # ID ist 1-based, Index ist 0-based
        
        if question_idx < 0 or question_idx >= len(questions):
            logger.error(f"Ungültige Frage ID: {question_id}")
            return False
        
        question = questions[question_idx]
        correct_index = question.get("correct_index", 0)
        
        # Speichere Antwort
        self.user_answers[question_id] = answer_index
        question["user_answer"] = answer_index
        question["attempts"] = question.get("attempts", 0) + 1
        
        # Überprüfe ob korrekt
        is_correct = answer_index == correct_index
        question["is_correct"] = is_correct
        
        # Wenn falsch, merke für Wiederholung
        if not is_correct:
            if question_id not in self.failed_questions:
                self.failed_questions.append(question_id)
            logger.info(f"Frage {question_id}: FALSCH (Versuch {question['attempts']})")
            return False
        else:
            # Wenn richtig nach falscher Antwort, entferne aus failed_questions
            if question_id in self.failed_questions:
                self.failed_questions.remove(question_id)
            logger.info(f"Frage {question_id}: RICHTIG")
            return True
    
    def next_question(self) -> Dict[str, Any]:
        """
        Geht zur nächsten Frage.
        
        Returns:
            Nächste Frage oder None wenn Quiz beendet
        """
        
        if not self.current_quiz:
            return None
        
        current_idx = self.current_quiz.get("current_question_index", 0)
        questions = self.current_quiz.get("questions", [])
        
        # Wenn failed_questions existieren, gehe zu diesen zurück
        failed = self.current_quiz.get("failed_questions", [])
        if failed:
            next_failed = [f for f in failed if f > (current_idx + 1)]
            if not next_failed:
                # Alle fehlgeschlagen beantwortet, Quiz vorbei
                self.completion_time = datetime.now()
                return None
            
            next_idx = next_failed[0] - 1
            self.current_quiz["current_question_index"] = next_idx
        else:
            # Keine fehlgeschlagenen, gehe zur nächsten
            next_idx = current_idx + 1
            self.current_quiz["current_question_index"] = next_idx
        
        if next_idx < len(questions):
            logger.info(f"Gehe zu Frage {next_idx + 1}")
            return questions[next_idx]
        
        self.completion_time = datetime.now()
        return None
    
    def get_quiz_statistics(self) -> Dict[str, Any]:
        """
        Berechnet Quiz-Statistiken.
        
        Returns:
            Statistik-Dict mit Ergebnissen
        """
        
        if not self.current_quiz:
            return None
        
        questions = self.current_quiz.get("questions", [])
        
        total = len(questions)
        correct = sum(1 for q in questions if q.get("is_correct"))
        incorrect = total - correct
        
        statistics = {
            "total": total,
            "correct": correct,
            "incorrect": incorrect,
            "percentage": round((correct / total * 100) if total > 0 else 0, 1),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "completion_time": self.completion_time.isoformat() if self.completion_time else None,
            "failed_questions": self.failed_questions
        }
        
        logger.info(f"Quiz-Statistiken: {correct}/{total} richtig ({statistics['percentage']}%)")
        return statistics
    
    def is_quiz_complete(self) -> bool:
        """
        Prüft ob das Quiz komplett beantwortet wurde.
        
        Returns:
            True wenn alle Fragen beantwortet und keine fehlgeschlagenen mehr
        """
        
        if not self.current_quiz:
            return False
        
        questions = self.current_quiz.get("questions", [])
        
        # Alle müssen beantwortet werden
        answered = sum(1 for q in questions if q.get("is_correct") is not None)
        
        if answered < len(questions):
            return False
        
        # Keine fehlgeschlagenen Fragen sollten übrig sein
        failed = [q["id"] for q in questions if q.get("is_correct") == False]
        
        return len(failed) == 0
    
    def export_quiz(self) -> Dict[str, Any]:
        """
        Exportiert das aktuelle Quiz mit allen Antworten und Statistiken.
        
        Returns:
            Quiz-Export-Struktur
        """
        
        if not self.current_quiz:
            logger.error("Kein aktives Quiz zum Exportieren")
            return None
        
        statistics = self.get_quiz_statistics()
        
        export_data = {
            "quiz": {
                "generated_at": datetime.now().isoformat(),
                "source_pdfs": self.current_quiz.get("source_pdfs", []),
                "theme": self.current_quiz.get("theme", ""),
                "total_questions": self.current_quiz.get("total_questions", 0),
                "questions": [
                    {
                        "id": q["id"],
                        "question": q["question"],
                        "options": q["options"],
                        "correct_index": q["correct_index"],
                        "user_answer": q.get("user_answer"),
                        "is_correct": q.get("is_correct"),
                        "difficulty": q.get("difficulty", "medium"),
                        "attempts": q.get("attempts", 0),
                        "source_chunk": q.get("source_chunk", "")
                    }
                    for q in self.current_quiz.get("questions", [])
                ],
                "statistics": statistics
            }
        }
        
        logger.info("Quiz erfolgreich exportiert")
        return export_data
    
    def import_quiz(self, import_data: Dict[str, Any]) -> bool:
        """
        Importiert ein vorher exportiertes Quiz.
        
        Args:
            import_data: Quiz-Import-Daten
            
        Returns:
            True wenn erfolgreich
        """
        
        try:
            if not import_data or "quiz" not in import_data:
                logger.error("Ungültige Import-Daten")
                return False
            
            quiz_data = import_data["quiz"]
            
            # Rekonstruiere Quiz-Objekt
            questions = []
            for q in quiz_data.get("questions", []):
                question_obj = {
                    "id": q["id"],
                    "question": q["question"],
                    "options": q["options"],
                    "correct_index": q["correct_index"],
                    "difficulty": q.get("difficulty", "medium"),
                    "user_answer": q.get("user_answer"),
                    "is_correct": q.get("is_correct"),
                    "attempts": q.get("attempts", 0),
                    "source_chunk": q.get("source_chunk", "")
                }
                questions.append(question_obj)
            
            self.current_quiz = {
                "source_pdfs": quiz_data.get("source_pdfs", []),
                "theme": quiz_data.get("theme", ""),
                "total_questions": quiz_data.get("total_questions", 0),
                "questions": questions,
                "current_question_index": 0,
                "failed_questions": quiz_data.get("statistics", {}).get("failed_questions", [])
            }
            
            logger.info(f"Quiz erfolgreich importiert ({len(questions)} Fragen)")
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Quiz-Import: {e}", exc_info=True)
            return False
    
    def get_failed_questions(self) -> List[Dict[str, Any]]:
        """
        Gibt die fehlgeschlagenen Fragen zurück.
        
        Returns:
            Liste der falsch beantworteten Fragen
        """
        
        if not self.current_quiz:
            return []
        
        questions = self.current_quiz.get("questions", [])
        return [q for q in questions if q["id"] in self.failed_questions]
    
    def reset_quiz(self):
        """Setzt das Quiz zurück."""
        self.current_quiz = None
        self.user_answers = {}
        self.failed_questions = []
        self.start_time = None
        self.completion_time = None
        logger.info("Quiz zurückgesetzt")
