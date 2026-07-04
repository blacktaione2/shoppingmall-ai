/*
 * chat-common.js 로직 검증 하니스
 * - window.location.hostname, fetch 를 mock 으로 주입
 * - 실제 chat-common.js 파일 내용을 평가한 뒤 전역 함수들을 테스트
 */
const fs = require("fs");
const vm = require("vm");
const path = require("path");

// chat-common.js 위치.
// 기본값: 이 프로젝트(shoppingmall_ai)와 Spring Boot 프로젝트(shoppingmall)가
//        같은 상위 폴더에 나란히 있다고 가정한 경로.
//   부모폴더/
//   ├── shoppingmall_ai/   (이 프로젝트, tests/ 가 여기)
//   └── shoppingmall/      (Spring Boot)
// 다른 위치라면 환경변수로 덮어쓰기: CHAT_COMMON_JS=/경로/chat-common.js node tests/test_chat_common.js
const CHAT_COMMON_JS =
  process.env.CHAT_COMMON_JS ||
  path.join(
    __dirname,
    "../../shoppingmall/src/main/resources/static/js/chat-common.js"
  );

const code = fs.readFileSync(CHAT_COMMON_JS, "utf-8");

// ── mock 상태 캡처 ──
let lastFetch = null;
const mockResponses = {
  "/chat/ask": { answer: "테스트 답변", intent: "SEMANTIC_SEARCH", confidence: 0.91 },
  "/chat/faq": [{ faq_id: 1, question: "배송 얼마나?", answer: "2~3일", category: "배송" }],
};

function makeFetch() {
  return function (url, opts) {
    lastFetch = { url: url, opts: opts || null };
    // path 매칭 (쿼리스트링 제거)
    const path = url.replace("http://localhost:8000", "").split("?")[0];
    const body = mockResponses[path];
    return Promise.resolve({
      ok: true,
      status: 200,
      json: function () {
        return Promise.resolve(body);
      },
    });
  };
}

// ── 평가 컨텍스트 구성 (window/fetch 주입) ──
const sandbox = {
  window: { location: { hostname: "localhost" } },
  fetch: makeFetch(),
  console: console,
};
vm.createContext(sandbox);
vm.runInContext(code, sandbox);

// ── 테스트 러너 ──
let passed = 0,
  failed = 0;
function check(name, cond, extra) {
  if (cond) {
    passed++;
    console.log("✅ " + name);
  } else {
    failed++;
    console.log("❌ " + name + (extra ? "  → " + extra : ""));
  }
}

(async function () {
  // 1) FASTAPI_BASE_URL 자동 구성 (const 는 컨텍스트 내 평가로 확인)
  const baseUrl = vm.runInContext("FASTAPI_BASE_URL", sandbox);
  check(
    "FASTAPI_BASE_URL = http://localhost:8000",
    baseUrl === "http://localhost:8000",
    baseUrl
  );

  // 2) MEMBER_ID 고정
  const memberId = vm.runInContext("MEMBER_ID", sandbox);
  check("MEMBER_ID === 1", memberId === 1, String(memberId));

  // 3) intentLabel — 6종 + fallback
  const labelCases = {
    STRUCTURED_QUERY: "검색결과",
    SEMANTIC_SEARCH: "추천",
    FAQ: "FAQ",
    ORDER_INQUIRY: "주문조회",
    COMPLAINT: "상담",
    SMALL_TALK: "잡담",
  };
  let labelOk = true;
  for (const k in labelCases) {
    if (sandbox.intentLabel(k) !== labelCases[k]) {
      labelOk = false;
      console.log("    라벨 불일치:", k, "→", sandbox.intentLabel(k));
    }
  }
  check("intentLabel 6종 매핑", labelOk);
  check(
    "intentLabel 미매핑 코드는 원문 유지",
    sandbox.intentLabel("UNKNOWN_X") === "UNKNOWN_X"
  );
  check("intentLabel(undefined) → '응답'", sandbox.intentLabel(undefined) === "응답");

  // 4) intentBadgeClass — 소문자 + default
  check(
    "intentBadgeClass(SEMANTIC_SEARCH)=badge--semantic_search",
    sandbox.intentBadgeClass("SEMANTIC_SEARCH") === "badge--semantic_search",
    sandbox.intentBadgeClass("SEMANTIC_SEARCH")
  );
  check(
    "intentBadgeClass(null)=badge--default",
    sandbox.intentBadgeClass(null) === "badge--default"
  );

  // 5) askChat — 요청 body/헤더/URL + 응답 파싱
  const askResult = await sandbox.askChat("운동화 추천해줘");
  check("askChat URL = /chat/ask", lastFetch.url === "http://localhost:8000/chat/ask", lastFetch.url);
  check("askChat method = POST", lastFetch.opts.method === "POST");
  check(
    "askChat Content-Type = application/json",
    lastFetch.opts.headers["Content-Type"] === "application/json"
  );
  const sentBody = JSON.parse(lastFetch.opts.body);
  check(
    "askChat body = {member_id:1, question:'운동화 추천해줘'}",
    sentBody.member_id === 1 && sentBody.question === "운동화 추천해줘",
    JSON.stringify(sentBody)
  );
  check(
    "askChat 응답 파싱 {answer,intent,confidence}",
    askResult.answer === "테스트 답변" &&
      askResult.intent === "SEMANTIC_SEARCH" &&
      askResult.confidence === 0.91
  );

  // 6) fetchFaqList — 카테고리 유무에 따른 URL
  await sandbox.fetchFaqList();
  check("fetchFaqList() URL = /chat/faq", lastFetch.url === "http://localhost:8000/chat/faq", lastFetch.url);
  await sandbox.fetchFaqList("배송");
  check(
    "fetchFaqList('배송') URL 에 category 인코딩 포함",
    lastFetch.url === "http://localhost:8000/chat/faq?category=" + encodeURIComponent("배송"),
    lastFetch.url
  );

  // 7) HTTP 오류 시 throw
  sandbox.fetch = function () {
    return Promise.resolve({ ok: false, status: 500, json: () => Promise.resolve({}) });
  };
  let threw = false;
  try {
    await sandbox.askChat("x");
  } catch (e) {
    threw = true;
  }
  check("askChat HTTP 500 시 예외 throw", threw);

  console.log("\n=== 결과: " + passed + " PASSED / " + failed + " FAILED ===");
  process.exit(failed === 0 ? 0 : 1);
})();
