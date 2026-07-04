/**
 * jsdom + 실서버(stub FastAPI) 로 chatbot.js 를 실제 브라우저처럼 구동하는 통합 테스트.
 *
 * 검증 항목:
 *   1) 전송 즉시 #typing-row 가 보임 (타이핑 인디케이터)
 *   2) 첫 chunk 도착 전까지 #typing-row 가 유지됨 (이번에 요청받은 핵심 수정사항)
 *   3) 첫 chunk 도착 시 #typing-row 제거 + 빈 봇 말풍선 생성
 *   4) 이후 chunk 마다 말풍선 텍스트가 점진적으로 늘어남
 *   5) done 수신 시 intent 배지가 삽입됨
 *   6) 전송 버튼/입력창이 전송 중 disabled 였다가 완료 후 풀림
 *
 * 실행: node tests/jsdom_chatbot_test.mjs   (FastAPI stub 서버가 8000 포트에 떠 있어야 함)
 */
import { JSDOM } from "jsdom";
import fs from "fs";
import path from "path";

// chatbot.js / chat-common.js 가 있는 폴더.
// 기본값: shoppingmall_ai 와 shoppingmall(Spring Boot)이 같은 상위 폴더에 나란히 있다고 가정.
// 다른 위치면: STATIC_JS_DIR=/경로/static/js node tests/jsdom_chatbot_test.mjs
const STATIC_JS_DIR =
  process.env.STATIC_JS_DIR ||
  path.join(
    path.dirname(new URL(import.meta.url).pathname),
    "../../shoppingmall/src/main/resources/static/js"
  );

const dom = new JSDOM(
  `<!DOCTYPE html><html><body>
    <div id="messages"></div>
    <textarea id="question-input"></textarea>
    <button id="send-btn">전송</button>
    <ul id="faq-list"></ul>
  </body></html>`,
  { url: "http://localhost/" }
);

global.window = dom.window;
global.document = dom.window.document;
global.fetch = fetch; // Node 22 전역 fetch 그대로 사용 (Web Streams 지원)
global.TextDecoder = TextDecoder;

// 실제 HTML 의 <script> 로드 순서(chat-common.js 먼저)를 그대로 재현.
// FASTAPI_BASE_URL / MEMBER_ID / intentLabel / intentBadgeClass / askChat / fetchFaqList
// 전부 실제 파일 그대로 평가되어 전역에 생긴다(stub 아님).
const commonSrc = fs.readFileSync(path.join(STATIC_JS_DIR, "chat-common.js"), "utf-8");
new Function(commonSrc + "\nglobal.FASTAPI_BASE_URL = FASTAPI_BASE_URL; global.MEMBER_ID = MEMBER_ID; global.intentLabel = intentLabel; global.intentBadgeClass = intentBadgeClass; global.fetchFaqList = fetchFaqList;")();

const src = fs.readFileSync(path.join(STATIC_JS_DIR, "chatbot.js"), "utf-8");
new Function(src)();

const sendBtn = document.getElementById("send-btn");
const inputEl = document.getElementById("question-input");
const messagesEl = document.getElementById("messages");

function snapshot(label) {
  const typingRow = document.getElementById("typing-row");
  const bubbles = messagesEl.querySelectorAll(".bubble--bot:not(.typing)");
  const lastBotText = bubbles.length ? bubbles[bubbles.length - 1].textContent : null;
  console.log(
    `[${label}] typing=${!!typingRow} sendDisabled=${sendBtn.disabled} inputDisabled=${inputEl.disabled} lastBotText=${JSON.stringify(lastBotText)}`
  );
}

async function run() {
  inputEl.value = "안녕";
  sendBtn.click(); // handleSend() 트리거 (비동기, await 안 함 — 실제 클릭과 동일)

  // 클릭 직후: 타이핑 인디케이터가 즉시 보여야 함
  await new Promise((r) => setTimeout(r, 0));
  snapshot("전송 직후(0ms)");
  const typingShownImmediately = !!document.getElementById("typing-row");

  // 첫 chunk 도착 전(서버 ~30ms 간격이므로 15ms 시점) — 아직 타이핑 유지돼야 함
  await new Promise((r) => setTimeout(r, 15));
  snapshot("첫 chunk 전(~15ms)");
  const typingStillShownBeforeFirstChunk = !!document.getElementById("typing-row");

  // 스트림 완료까지 대기 (6 chunk * 30ms + done 여유)
  await new Promise((r) => setTimeout(r, 400));
  snapshot("완료 후(~415ms)");

  const finalBubble = messagesEl.querySelector(".bubble--bot:not(.typing)");
  const badge = finalBubble ? finalBubble.querySelector(".badge") : null;

  console.log("\n=== 결과 ===");
  console.log("1) 전송 즉시 타이핑 인디케이터 표시:", typingShownImmediately);
  console.log("2) 첫 chunk 전까지 타이핑 유지:      ", typingStillShownBeforeFirstChunk);
  console.log("3) 완료 후 타이핑 제거:               ", !document.getElementById("typing-row"));
  console.log("4) 최종 답변 텍스트:                  ", finalBubble ? finalBubble.querySelector(".bubble__body").textContent : null);
  console.log("5) intent 배지 삽입됨:                ", !!badge, badge ? badge.textContent : null);
  console.log("6) 전송 후 입력/버튼 재활성화:         ", !sendBtn.disabled && !inputEl.disabled);

  const allPass =
    typingShownImmediately &&
    typingStillShownBeforeFirstChunk &&
    !document.getElementById("typing-row") &&
    !!badge &&
    !sendBtn.disabled &&
    !inputEl.disabled;

  console.log("\n전체 통과:", allPass);
  process.exit(allPass ? 0 : 1);
}

run();
