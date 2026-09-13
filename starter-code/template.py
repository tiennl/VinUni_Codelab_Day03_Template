"""
Lab #3: Baseline Chatbot vs ReAct Agent
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.
"""

import json
import re
from tools import TOOL_DEFINITIONS, TOOL_MAP, get_flight_info, get_weather_forecast

SYSTEM_PROMPT = """Bạn là một ReAct Agent thông minh hỗ trợ khách hàng Vingroup.
Bạn chỉ sử dụng các công cụ sau:
{tools}

Quy trình trả lời bắt buộc:
Thought: <Suy nghĩ bước tiếp theo>
Action: {{"name": "<tên tool>", "args": {{<tham số>}}}}
Observation: <Kết quả từ tool>
... (Lặp lại cho tới khi có đủ dữ liệu)
Final Answer: <Câu trả lời hoàn chỉnh cho khách hàng>
"""

# Mã sân bay hỗ trợ và các tên gọi thường gặp trong câu hỏi của khách.
CITY_ALIASES = {
    "HAN": ["hà nội", "ha noi", "hanoi"],
    "SGN": ["tp. hồ chí minh", "hồ chí minh", "sài gòn", "sai gon", "saigon"],
    "DAD": ["đà nẵng", "da nang", "danang"],
}

FLIGHT_KEYWORDS = ["chuyến bay", "vé máy bay", "vé bay", "đặt vé", "flight", "bay từ"]
WEATHER_KEYWORDS = ["thời tiết", "mặc gì", "nên mặc", "nhiệt độ", "weather", "mưa", "nắng"]
BRANDS = ["Vinpearl", "VinFast", "Vinhomes", "Vinmec", "Vingroup"]

PRICE_UNITS = {"triệu": 1_000_000, "tr": 1_000_000, "nghìn": 1_000, "ngàn": 1_000, "k": 1_000}


def _format_vnd(amount: int) -> str:
    return f"{amount:,}".replace(",", ".") + "đ"


class ChatbotBaseline:
    """Baseline LLM Chatbot (Không sử dụng ReAct Loop hay Tools)"""

    def query(self, user_input: str) -> dict:
        """Trả lời bằng 1 lượt sinh văn bản duy nhất, không truy cập dữ liệu thật.

        Đây chính là điểm yếu cần quan sát ở Milestone 1: chatbot không có
        `tool_calls` nào nên chỉ có thể nói chung chung hoặc bịa thông tin.
        """
        answer = (
            f"[Chatbot Baseline] Về câu hỏi \"{user_input}\": mình chỉ có thể trả lời "
            "dựa trên kiến thức đã học, không tra cứu được dữ liệu chuyến bay hay thời "
            "tiết theo thời gian thực. Bạn vui lòng kiểm tra lại trên website/ứng dụng "
            "để có thông tin chính xác nhé."
        )
        return {
            "status": "success",
            "answer": answer,
            "tool_calls": [],
        }


class ReActAgent:
    """ReAct Agent có sử dụng Thought-Action-Observation Loop"""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace = []

    # ------------------------------------------------------------------
    # Phần "suy luận": ở lab này ta mô phỏng LLM bằng luật đơn giản.
    # Kết quả trả về là các dòng Action dạng chuỗi JSON, đúng như định dạng
    # mà SYSTEM_PROMPT yêu cầu LLM sinh ra.
    # ------------------------------------------------------------------
    def _detect_city_codes(self, user_input: str) -> list:
        lowered = user_input.lower()
        hits = []
        for match in re.finditer(r"\b(HAN|SGN|DAD)\b", user_input.upper()):
            hits.append((match.start(), match.group(1)))
        for code, aliases in CITY_ALIASES.items():
            for alias in aliases:
                pos = lowered.find(alias)
                if pos != -1:
                    hits.append((pos, code))
                    break
        hits.sort(key=lambda item: item[0])

        ordered = []
        for _, code in hits:
            if code not in ordered:
                ordered.append(code)
        return ordered

    def _detect_max_price(self, user_input: str) -> int:
        match = re.search(
            r"(\d+(?:[.,]\d+)?)\s*(triệu|tr\b|nghìn|ngàn|k\b)", user_input.lower()
        )
        if not match:
            return 5_000_000
        amount = float(match.group(1).replace(",", "."))
        unit = match.group(2).strip()
        return int(amount * PRICE_UNITS[unit])

    def _plan(self, user_input: str) -> list:
        """Sinh danh sách Action (chuỗi JSON) cần thực hiện cho câu hỏi."""
        lowered = user_input.lower()
        codes = self._detect_city_codes(user_input)
        actions = []

        wants_flight = any(kw in lowered for kw in FLIGHT_KEYWORDS)
        wants_weather = any(kw in lowered for kw in WEATHER_KEYWORDS)

        # Chỉ gọi tool tìm chuyến bay khi xác định được cả điểm đi lẫn điểm đến.
        if wants_flight and len(codes) >= 2:
            actions.append(
                json.dumps(
                    {
                        "name": "get_flight_info",
                        "args": {
                            "origin": codes[0],
                            "destination": codes[1],
                            "max_price": self._detect_max_price(user_input),
                        },
                    },
                    ensure_ascii=False,
                )
            )

        if wants_weather and codes:
            actions.append(
                json.dumps(
                    {"name": "get_weather_forecast", "args": {"city_code": codes[-1]}},
                    ensure_ascii=False,
                )
            )

        return actions

    def _parse_action(self, action_str: str):
        """Trap 2: LLM hay trả Action sai định dạng -> luôn bọc trong try/except."""
        try:
            action = json.loads(action_str)
        except (json.JSONDecodeError, TypeError):
            return None, "Invalid JSON format"

        if not isinstance(action, dict) or "name" not in action:
            return None, "Invalid JSON format"

        # Trap 1: chuẩn hoá tên tool trước khi tra cứu trong TOOL_MAP.
        name = str(action["name"]).strip().lower()
        args = action.get("args", {})
        if not isinstance(args, dict):
            return None, "Invalid args format"
        if name not in TOOL_MAP:
            return None, f"Unknown tool: {name}"
        return {"name": name, "args": args}, None

    def _execute(self, name: str, args: dict):
        try:
            return TOOL_MAP[name](**args)
        except TypeError as exc:
            return {"error": f"Invalid arguments for {name}: {exc}"}

    # ------------------------------------------------------------------
    # Sinh câu trả lời từ các Observation thu được
    # ------------------------------------------------------------------
    def _describe_flights(self, args: dict, flights) -> str:
        origin = args.get("origin", "?")
        destination = args.get("destination", "?")
        budget = _format_vnd(int(args.get("max_price", 5_000_000)))
        if not flights:
            return (
                f"Rất tiếc, hiện không có chuyến bay nào từ {origin} đi {destination} "
                f"với mức giá dưới {budget}. Bạn thử nới ngân sách hoặc đổi ngày bay nhé."
            )
        items = "; ".join(
            f"{fl['flight_number']} ({fl['airline']}) khởi hành {fl['departure_time']} "
            f"giá {_format_vnd(fl['price_vnd'])}"
            for fl in flights
        )
        return (
            f"Có {len(flights)} chuyến bay {origin} → {destination} dưới {budget}: {items}."
        )

    def _describe_weather(self, args: dict, weather) -> str:
        if not isinstance(weather, dict) or "error" in weather:
            return (
                f"Chưa lấy được dữ liệu thời tiết cho {args.get('city_code', '?')}. "
                "Bạn kiểm tra lại mã thành phố giúp mình nhé."
            )
        return (
            f"Thời tiết {weather['city']}: {weather['temperature_c']}°C, "
            f"{weather['condition']}, độ ẩm {weather['humidity_pct']}%. "
            f"Gợi ý trang phục: {weather['recommendation']}"
        )

    def _answer_from(self, observations: list) -> str:
        parts = []
        for name, args, result in observations:
            if name == "get_flight_info":
                parts.append(self._describe_flights(args, result))
            elif name == "get_weather_forecast":
                parts.append(self._describe_weather(args, result))
        return " ".join(parts)

    def _answer_without_tools(self, user_input: str) -> str:
        """Câu hỏi FAQ không cần tool -> trả lời trực tiếp trong 1 bước."""
        subject = next((b for b in BRANDS if b.lower() in user_input.lower()), "Vingroup")
        return (
            f"Câu hỏi này không cần tra cứu chuyến bay hay thời tiết. Với dịch vụ {subject}, "
            "chính sách đổi/trả vé phụ thuộc vào hạng vé bạn đã mua: vé linh hoạt được đổi "
            "miễn phí trước giờ khởi hành, vé tiết kiệm thường chịu phí đổi và không hoàn. "
            f"Bạn vui lòng liên hệ tổng đài {subject} để được hỗ trợ theo đúng mã đặt chỗ."
        )

    # ------------------------------------------------------------------
    # ReAct Loop
    # ------------------------------------------------------------------
    def run(self, user_input: str) -> dict:
        self.trace = []
        observations = []
        iteration = 0

        pending_actions = self._plan(user_input)

        for action_str in pending_actions:
            # Milestone 4: safeguard chống lặp vô tận.
            if iteration >= self.max_iterations:
                return self._max_iterations_result(iteration)

            iteration += 1
            action, error = self._parse_action(action_str)
            if error:
                self.trace.append(
                    {
                        "iteration": iteration,
                        "thought": "Action vừa sinh ra không hợp lệ, cần thử lại.",
                        "action": action_str,
                        "observation": f"Invalid JSON format ({error})",
                    }
                )
                continue

            result = self._execute(action["name"], action["args"])
            self.trace.append(
                {
                    "iteration": iteration,
                    "thought": f"Cần gọi {action['name']} để lấy dữ liệu thực tế.",
                    "action": action,
                    "observation": result,
                }
            )
            observations.append((action["name"], action["args"], result))

        if not observations:
            # Không cần tool: chốt Final Answer ngay trong bước đầu tiên.
            iteration += 1
            answer = self._answer_without_tools(user_input)
            self.trace.append(
                {
                    "iteration": iteration,
                    "thought": "Câu hỏi không cần tool, trả lời trực tiếp.",
                    "action": None,
                    "final_answer": answer,
                }
            )
            return self._completed_result(answer, iteration)

        if len(observations) == 1:
            # Một Observation duy nhất -> map thẳng thành câu trả lời, không tốn thêm bước.
            answer = self._answer_from(observations)
            self.trace[-1]["final_answer"] = answer
            return self._completed_result(answer, iteration)

        # Nhiều Observation -> cần thêm một bước tổng hợp trước khi trả lời.
        if iteration >= self.max_iterations:
            return self._max_iterations_result(iteration)

        iteration += 1
        answer = self._answer_from(observations)
        self.trace.append(
            {
                "iteration": iteration,
                "thought": "Đã đủ dữ liệu từ các tool, tổng hợp lại cho khách hàng.",
                "action": None,
                "final_answer": answer,
            }
        )
        return self._completed_result(answer, iteration)

    def _completed_result(self, answer: str, iteration: int) -> dict:
        return {
            "status": "completed",
            "answer": answer,
            "iterations": iteration,
            "trace": self.trace,
        }

    def _max_iterations_result(self, iteration: int) -> dict:
        return {
            "status": "max_iterations_reached",
            "answer": "Không thể hoàn thành trong số bước tối đa.",
            "iterations": iteration,
            "trace": self.trace,
        }


def main():
    user_query = "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, rồi cho biết thời tiết SGN nên mặc gì?"

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query)["answer"])

    print("\n=== RUNNING REACT AGENT ===")
    agent = ReActAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Status:", result["status"], "| Iterations:", result["iterations"])
    print("Answer:", result["answer"])
    print("Trace Log:", json.dumps(result["trace"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
