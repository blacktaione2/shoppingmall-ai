"""
services/notification_service.py
환불 신청 등 사람의 확인이 필요한 이벤트를 관리자에게 이메일로 알린다.

[범위]
- 실제 ORDERS 상태 변경/환불 처리는 여전히 Spring Boot 책임 영역이다
  (graph/tools.py request_refund 주석 참고). 이 모듈은 "관리자에게 알림만"
  보낸다 — DB 쓰기 없음.
- FastAPI 에는 기존에 이메일 발송 인프라가 없어서(Gmail SMTP는 Spring Boot
  EmailService 전용), 표준 라이브러리 smtplib 로 최소 구현한다(새 의존성 없음).

[동작 — best-effort]
메일 발송 실패(SMTP 미설정/네트워크 오류 등)는 로그만 남기고 예외를 올리지
않는다. save_chat_history/saveChatToken 과 동일한 가용성 우선 폴백 철학 —
알림 실패가 사용자에게 보이는 환불 접수 응답을 막아서는 안 된다.

[설정 — .env]
ADMIN_ALERT_EMAIL : 알림을 받을 관리자 이메일. 비어있으면 발송 생략.
MAIL_USERNAME/MAIL_PASSWORD : Gmail SMTP 계정(Spring Boot와 동일 계정 재사용 가능,
                              앱 비밀번호 필요).
"""
import asyncio
import logging
import os
import smtplib
from email.message import EmailMessage

logger = logging.getLogger(__name__)

_SMTP_HOST = "smtp.gmail.com"
_SMTP_PORT = 587


def _send_refund_admin_email_sync(order_id: str, member_id: int, reason: str | None) -> None:
    admin_email = os.getenv("ADMIN_ALERT_EMAIL", "").strip()
    mail_user = os.getenv("MAIL_USERNAME", "").strip()
    mail_pwd = os.getenv("MAIL_PASSWORD", "").strip()
    if not admin_email or not mail_user or not mail_pwd:
        logger.warning(
            "환불 알림 메일 생략 — ADMIN_ALERT_EMAIL/MAIL_USERNAME/MAIL_PASSWORD 미설정"
        )
        return

    msg = EmailMessage()
    msg["Subject"] = f"[쇼핑몰] 챗봇 환불 신청 접수 - 주문 {order_id}"
    msg["From"] = mail_user
    msg["To"] = admin_email
    body = (
        f"AI 챗봇을 통해 환불 신청이 접수됐습니다.\n\n"
        f"주문번호: {order_id}\n"
        f"회원 ID: {member_id}\n"
        f"사유: {reason or '(미입력)'}\n\n"
        "실제 환불 처리는 쇼핑몰 관리자 화면에서 진행해 주세요."
    )
    msg.set_content(body)

    with smtplib.SMTP(_SMTP_HOST, _SMTP_PORT, timeout=10) as server:
        server.starttls()
        server.login(mail_user, mail_pwd)
        server.send_message(msg)


async def send_refund_admin_email(order_id: str, member_id: int, reason: str | None) -> None:
    """환불 신청 접수를 관리자에게 이메일로 알린다(best-effort, 실패해도 예외 안 던짐)."""
    try:
        await asyncio.to_thread(_send_refund_admin_email_sync, order_id, member_id, reason)
    except Exception:
        logger.exception(
            "환불 알림 메일 발송 실패(무시): order_id=%s member_id=%s", order_id, member_id
        )
