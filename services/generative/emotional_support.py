"""Assistente de suporte emocional baseado em regras simples."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SupportDocument:
    """Representa um conteúdo breve de apoio emocional."""

    topic: str
    summary: str
    keywords: List[str]


class EmotionalSupportAI:
    """Assistente rule-based para fornecer orientações de bem-estar."""

    def __init__(self):
        self.documents = self._load_documents()
        self.conversation_history: Dict[int, List[Dict[str, str]]] = {}
        logger.info("Assistente de suporte emocional inicializado.")

    def _load_documents(self) -> List[SupportDocument]:
        """Carrega conhecimento básico de suporte emocional."""
        return [
            SupportDocument(
                topic="burnout",
                summary="Sinais de burnout incluem exaustão, irritabilidade e perda de foco. Faça pausas, converse com alguém de confiança e procure apoio especializado quando necessário.",
                keywords=["cansado", "exausto", "burnout", "sobrecarga", "sobrecarregado"],
            ),
            SupportDocument(
                topic="mindfulness",
                summary="Práticas simples de respiração, meditação guiada ou registrar pensamentos ajudam a reduzir ansiedade e clarear ideias.",
                keywords=["mindfulness", "respiração", "meditar", "ansioso", "ansiedade"],
            ),
            SupportDocument(
                topic="limites",
                summary="Defina limites claros de trabalho e reservas de descanso. Ajustar expectativas com a equipe evita sobrecarga constante.",
                keywords=["limite", "não consigo", "pressão", "deadline", "sobrecarga"],
            ),
            SupportDocument(
                topic="sono",
                summary="Manter rotina de sono entre 7 e 9 horas fortalece energia e humor. Evite telas e cafeína antes de dormir.",
                keywords=["sono", "insônia", "dormir", "cansaço"],
            ),
            SupportDocument(
                topic="apoio_social",
                summary="Converse com pessoas de confiança e compartilhe como se sente. Apoio social genuíno é fator de proteção emocional.",
                keywords=["sozinho", "isolado", "ninguém", "apoio"],
            ),
        ]

    def chat(self, user_id: int, message: str, context: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        """Gera resposta empática baseada em palavras-chave e contexto."""
        logger.info("Processando mensagem do usuário %s", user_id)
        history = self.conversation_history.setdefault(user_id, [])
        history.append({"timestamp": datetime.utcnow().isoformat(), "message": message})

        matched_docs = self._match_documents(message)
        response_parts: List[str] = []

        if context:
            response_parts.extend(self._contextual_notes(context))

        if matched_docs:
            response_parts.append("Aqui estão algumas orientações que podem ajudar:")
            for doc in matched_docs:
                response_parts.append(f"- *{doc.topic.capitalize()}*: {doc.summary}")
        else:
            response_parts.append("Percebo que você está passando por um momento desafiador. Tente identificar pequenas ações de autocuidado hoje, como uma pausa breve, respiração profunda ou conversar com alguém de confiança.")

        response_parts.append("Se sentir que a situação está impactando gravemente seu bem-estar, considere buscar ajuda profissional. Você não está sozinho.")

        response_text = "\n".join(response_parts)
        history.append({"timestamp": datetime.utcnow().isoformat(), "response": response_text})

        return {
            "response": response_text,
            "sources": [doc.topic for doc in matched_docs],
            "confidence": min(0.9, 0.6 + 0.1 * len(matched_docs)),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _match_documents(self, message: str) -> List[SupportDocument]:
        """Seleciona documentos cujos keywords aparecem na mensagem."""
        lower_msg = message.lower()
        matches = [doc for doc in self.documents if any(keyword in lower_msg for keyword in doc.keywords)]
        if not matches:
            logger.info("Nenhum tópico específico identificado, retornando mensagens padrão.")
        return matches

    def _contextual_notes(self, context: Dict[str, object]) -> List[str]:
        """Gera observações personalizadas a partir de contexto adicional."""
        notes: List[str] = []
        checkins = context.get("recent_checkins") or []
        if checkins:
            avg_stress = sum(entry.get("nivel_stress", 5) for entry in checkins) / len(checkins)
            notes.append(f"Você mencionou níveis médios de stress por volta de {avg_stress:.1f}/10 nas últimas interações.")
        fatigue = context.get("fatigue_level")
        if fatigue:
            notes.append(f"Nível de fadiga detectado: {fatigue}. Pequenas pausas e movimento leve podem ajudar.")
        wellbeing_score = context.get("wellbeing_score")
        if wellbeing_score is not None:
            notes.append(f"Score de bem-estar recente: {wellbeing_score}/100. Ajustes graduais podem fazer diferença.")
        return notes

    def clear_conversation(self, user_id: int) -> None:
        """Limpa histórico do usuário."""
        self.conversation_history.pop(user_id, None)
        logger.info("Histórico do usuário %s limpo.", user_id)


if __name__ == "__main__":
    assistant = EmotionalSupportAI()
    example = assistant.chat(
        user_id=1,
        message="Tenho me sentido muito cansado e ansioso com tantos prazos.",
        context={"recent_checkins": [{"nivel_stress": 8, "horas_trabalhadas": 10}], "wellbeing_score": 45},
    )
    print("Resposta:\n", example["response"])

