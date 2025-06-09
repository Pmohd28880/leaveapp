import logging
from openai import OpenAI
from dotenv import load_dotenv
import os
import json


# Set up independent logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('agent.log')
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def orchestrate_leave_approval(leave_request, team_members):
    try:
        logger.info("Starting leave approval process")

        # Step 1: HR Policy Check
        hr_prompt = {
            "role": "system",
            "content": """You are an HR agent. Evaluate leave requests against policy:
            1. Approve if: sick leave (< 5 days) OR vacation with notice.
            2. Deny if: reason is vague or violates policy.
            Respond with JSON format: {"status": "Approved|Denied", "reason": "string"}"""
        }

        user_prompt = {
            "role": "user",
            "content": f"Reason: {leave_request['reason']}. Dates: {leave_request['start_date']} to {leave_request['end_date']}"
        }

        logger.debug(f"Sending to OpenAI: {hr_prompt} | {user_prompt}")

        hr_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[hr_prompt, user_prompt],
            response_format={"type": "json_object"}
        )

        hr_decision = json.loads(hr_response.choices[0].message.content)
        logger.info(f"HR decision: {hr_decision}")

        if hr_decision.get('status', '').lower() == 'denied':
            return hr_decision

        # Step 2: Team Coverage Check (Fixed the f-string issue)
        team_prompt = {
            "role": "system",
            "content": f"""Team members: {len(team_members)}. Minimum required: 2.
            Respond with JSON format: {{"status": "Approved|Denied", "reason": "string"}}"""
        }

        logger.debug(f"Sending to OpenAI: {team_prompt}")

        team_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[team_prompt],
            response_format={"type": "json_object"}
        )

        team_decision = json.loads(team_response.choices[0].message.content)
        logger.info(f"Team decision: {team_decision}")

        return team_decision

    except Exception as e:
        logger.exception("Error in orchestrate_leave_approval:")
        return {"status": "Error", "reason": "System error occurred"}
