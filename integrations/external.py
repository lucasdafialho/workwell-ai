"""
Integrações externas: Slack, Teams, Wearables, etc.
"""

import httpx
import asyncio
import os
from typing import Dict, List, Optional
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SlackIntegration:
    """Integração com Slack para envio de insights."""
    
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
    
    async def send_insight(
        self,
        user_id: int,
        insight: Dict,
        channel: Optional[str] = None
    ) -> bool:
        """
        Envia insight diário personalizado via Slack.
        
        Args:
            user_id: ID do usuário
            insight: Dicionário com insights
            channel: Canal do Slack (opcional)
            
        Returns:
            True se enviado com sucesso
        """
        if not self.webhook_url:
            logger.warning("Webhook URL do Slack não configurado")
            return False
        
        message = {
            "text": f"Insights de Bem-estar - Usuário {user_id}",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "📊 Insights de Bem-estar"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Nível de Stress:* {insight.get('stress_level', 'N/A')}/10\n"
                               f"*Score de Bem-estar:* {insight.get('wellbeing_score', 'N/A')}/100\n"
                               f"*Recomendações:* {insight.get('recommendations', 'Nenhuma')}"
                    }
                }
            ]
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.webhook_url,
                    json=message,
                    timeout=10.0
                )
                response.raise_for_status()
                logger.info(f"Insight enviado ao Slack para usuário {user_id}")
                return True
        except Exception as e:
            logger.error(f"Erro ao enviar ao Slack: {e}")
            return False


class TeamsIntegration:
    """Integração com Microsoft Teams."""
    
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url or os.getenv("TEAMS_WEBHOOK_URL")
    
    async def send_alert(
        self,
        user_id: int,
        alert_type: str,
        message: str
    ) -> bool:
        """
        Envia alerta via Teams.
        
        Args:
            user_id: ID do usuário
            alert_type: Tipo de alerta ('warning', 'critical', 'info')
            message: Mensagem do alerta
            
        Returns:
            True se enviado com sucesso
        """
        if not self.webhook_url:
            return False
        
        color_map = {
            'warning': 'FFA500',
            'critical': 'FF0000',
            'info': '0078D4'
        }
        
        teams_message = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "summary": f"Alerta WorkWell - {alert_type}",
            "themeColor": color_map.get(alert_type, '0078D4'),
            "sections": [
                {
                    "activityTitle": f"Alerta de {alert_type.title()}",
                    "text": message,
                    "facts": [
                        {
                            "name": "Usuário",
                            "value": str(user_id)
                        },
                        {
                            "name": "Data",
                            "value": datetime.now().isoformat()
                        }
                    ]
                }
            ]
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.webhook_url,
                    json=teams_message,
                    timeout=10.0
                )
                response.raise_for_status()
                return True
        except Exception as e:
            logger.error(f"Erro ao enviar ao Teams: {e}")
            return False


class WearablesIntegration:
    """Integração com APIs de wearables para dados biométricos."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("WEARABLES_API_KEY")
        self.base_url = "https://api.wearables.example.com"  # Placeholder
    
    async def fetch_heart_rate_data(
        self,
        user_id: int,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict]:
        """Busca dados de batimento cardíaco.
        
        Args:
            user_id: ID do usuário
            start_date: Data inicial
            end_date: Data final
            
        Returns:
            Lista de registros de batimento cardíaco
        """
        logger.info(f"Buscando dados de batimento cardíaco para usuário {user_id}")
        
        return [
            {
                "timestamp": datetime.now().isoformat(),
                "heart_rate": 72,
                "resting_heart_rate": 65
            }
        ]
    
    async def fetch_sleep_data(
        self,
        user_id: int,
        date: datetime
    ) -> Dict:
        """Busca dados de sono.
        
        Args:
            user_id: ID do usuário
            date: Data
            
        Returns:
            Dicionário com dados de sono
        """
        logger.info(f"Buscando dados de sono para usuário {user_id}")
        
        return {
            "date": date.isoformat(),
            "total_sleep_hours": 7.5,
            "deep_sleep_hours": 2.0,
            "rem_sleep_hours": 1.5,
            "sleep_quality": 85
        }


class CalendarIntegration:
    """Integração com calendários para contexto temporal."""
    
    async def get_busy_periods(
        self,
        user_id: int,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict]:
        """Obtém períodos ocupados do calendário.
        
        Args:
            user_id: ID do usuário
            start_date: Data inicial
            end_date: Data final
            
        Returns:
            Lista de períodos ocupados
        """
        return [
            {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "title": "Reunião",
                "busy": True
            }
        ]


class ExternalIntegrations:
    """Classe principal para gerenciar todas as integrações externas."""
    
    def __init__(self):
        self.slack = SlackIntegration()
        self.teams = TeamsIntegration()
        self.wearables = WearablesIntegration()
        self.calendar = CalendarIntegration()
    
    async def send_daily_insights(
        self,
        user_id: int,
        insights: Dict,
        channels: List[str] = ["slack"]
    ):
        """
        Envia insights diários através de múltiplos canais.
        
        Args:
            user_id: ID do usuário
            insights: Dicionário com insights
            channels: Lista de canais para enviar
        """
        tasks = []
        
        if "slack" in channels:
            tasks.append(self.slack.send_insight(user_id, insights))
        
        if "teams" in channels:
            tasks.append(self.teams.send_alert(user_id, "info", str(insights)))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info(f"Insights enviados para usuário {user_id} através de {channels}")
        return results


if __name__ == "__main__":
    async def main():
        integrations = ExternalIntegrations()
        
        insights_example = {
            "stress_level": 7,
            "wellbeing_score": 65,
            "recommendations": "Considere fazer uma pausa"
        }
        
        await integrations.send_daily_insights(1, insights_example, channels=["slack"])

    asyncio.run(main())
