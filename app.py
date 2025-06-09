from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, login_user, login_required, current_user, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User, Leave, Team
from datetime import datetime, timedelta
import os
import logging
from logging.handlers import RotatingFileHandler
import json
from openai import OpenAI
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import List, Dict, Optional
import random
import re

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = "IAmherebaby"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///leave.db'
db.init_app(app)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Remove all existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
    handler.close()

# Create file handler
file_handler = RotatingFileHandler(
    'app.log',
    maxBytes=1024*1024,  # 1MB
    backupCount=5,
    encoding='utf-8'
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))

# Add handler to logger
logger.addHandler(file_handler)

# Also configure Flask's default logger
app.logger.handlers.clear()
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

# Flask-Login setup
login_manager = LoginManager(app)
login_manager.login_view = 'login'


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# =====================================================
# TRULY AGENTIC LEAVE MANAGEMENT SYSTEM - FIXED
# =====================================================

@dataclass
class LeaveRequest:
    reason: str
    start_date: str
    end_date: str
    user_id: int
    duration_days: int = None

    def __post_init__(self):
        if self.duration_days is None:
            start = datetime.strptime(self.start_date, '%Y-%m-%d')
            end = datetime.strptime(self.end_date, '%Y-%m-%d')
            self.duration_days = (end - start).days + 1


class AgenticLeaveSystem:
    def __init__(self):
        # Company policies as guidelines, not hard rules
        self.company_context = {
            'sick_leave_guidelines': 5,
            'vacation_notice_preference': 7,
            'max_standard_leave': 21,
            'min_team_size': 2,
            'company_values': ['employee_wellbeing', 'flexibility', 'team_collaboration'],
            'busy_periods': ['month_end', 'quarter_end', 'project_deadlines']
        }

        # Agent memory and learning
        self.decision_memory = []
        self.pattern_insights = {}

    def make_intelligent_decision(self, leave_request: LeaveRequest, team_members: List,
                                  additional_context: Dict = None) -> Dict:
        """
        Main agentic decision-making process using AI reasoning
        """
        try:
            logger.info(f"🤖 Starting agentic analysis for: {leave_request}")

            # Step 1: Gather all available context
            context = self._gather_comprehensive_context(
                leave_request, team_members, additional_context)

            # Step 2: AI-driven reasoning (this is the truly agentic part)
            reasoning_result = self._ai_reasoning_engine(
                leave_request, context)

            # Step 3: If AI suggests gathering more info, do it
            if reasoning_result.get('need_more_info'):
                additional_info = self._gather_additional_information(
                    reasoning_result['info_needed'])
                context.update(additional_info)
                # Re-analyze with new information
                reasoning_result = self._ai_reasoning_engine(
                    leave_request, context)

            # Step 4: Learn from this decision for future use
            self._update_agent_memory(leave_request, reasoning_result, context)

            return reasoning_result

        except Exception as e:
            logger.exception("Error in agentic decision making:")
            return {
                'status': 'Error',
                'reason': 'System encountered an error during analysis',
                'confidence': 0.0,
                'escalate': True,
                'agent_notes': f'Error: {str(e)}'
            }

    def _gather_comprehensive_context(self, leave_request: LeaveRequest,
                                      team_members: List, additional_context: Dict = None) -> Dict:
        """Gather all available context for decision making"""

        # Get user's leave history
        user_leave_history = self._get_user_leave_history(
            leave_request.user_id)

        # Analyze team workload (simulated - in real app, this would check actual data)
        team_workload = self._analyze_team_workload(
            team_members, leave_request.start_date)

        # Check for business critical periods
        business_impact = self._assess_business_impact(
            leave_request.start_date, leave_request.end_date)

        # Get similar past decisions for learning
        similar_cases = self._find_similar_past_decisions(leave_request)

        context = {
            'leave_request': {
                'reason': leave_request.reason,
                'duration': leave_request.duration_days,
                'dates': f"{leave_request.start_date} to {leave_request.end_date}",
                'user_id': leave_request.user_id
            },
            'team_context': {
                'total_members': len(team_members),
                'available_members': team_workload['available_count'],
                'team_workload': team_workload['workload_level']
            },
            'user_history': user_leave_history,
            'business_context': business_impact,
            'company_policies': self.company_context,
            'similar_past_decisions': similar_cases,
            'current_date': datetime.now().strftime('%Y-%m-%d')
        }

        if additional_context:
            context.update(additional_context)

        return context

    def _ai_reasoning_engine(self, leave_request: LeaveRequest, context: Dict) -> Dict:
        """
        The core AI reasoning engine - FIXED VERSION
        """
        try:
            # Create a comprehensive system prompt for true agentic behavior
            system_prompt = f"""
You are an intelligent HR decision-making agent with the ability to reason, learn, and make nuanced judgments about employee leave requests.

CORE CAPABILITIES:
- Analyze complex situations with multiple factors
- Balance company needs with employee wellbeing
- Make exceptions when justified by circumstances
- Learn from past decisions and patterns
- Escalate when human judgment is needed

COMPANY CONTEXT:
{json.dumps(context['company_policies'], indent=2)}

CURRENT SITUATION ANALYSIS:
Leave Request Details: {json.dumps(context['leave_request'], indent=2)}
Team Situation: {json.dumps(context['team_context'], indent=2)}
Employee History: {json.dumps(context['user_history'], indent=2)}
Business Impact: {json.dumps(context['business_context'], indent=2)}
Similar Past Decisions: {json.dumps(context['similar_past_decisions'], indent=2)}

DECISION FRAMEWORK:
1. Employee wellbeing is a top priority
2. Business continuity must be maintained
3. Exceptional circumstances may override standard policies
4. Transparency in decision-making is essential
5. Learning from each decision improves future outcomes

REASONING PROCESS:
1. Analyze the legitimacy and urgency of the request
2. Consider the employee's history and patterns
3. Evaluate business and team impact
4. Look for precedents in similar cases
5. Consider if any additional information is needed
6. Make a reasoned decision with clear justification

RESPONSE FORMAT - YOU MUST RESPOND WITH VALID JSON:
{{
    "status": "Approved|Denied|Escalate",
    "reason": "Detailed explanation of your reasoning process",
    "confidence": 0.85,
    "escalate": false,
    "need_more_info": false,
    "info_needed": [],
    "agent_reasoning": "Step-by-step thought process",
    "precedent_used": "Reference to similar past cases if applicable",
    "exception_made": "If standard policy was overridden, explain why",
    "learning_notes": "What this case teaches for future decisions"
}}

Think step by step and provide your reasoned decision as valid JSON:
"""

            user_prompt = f"""
Please analyze this leave request and provide your decision with detailed reasoning in JSON format.

Key factors to consider:
- This is a {leave_request.duration_days}-day {leave_request.reason} leave
- Standard policy suggests max {self.company_context['sick_leave_guidelines']} days for sick leave
- Team has {context['team_context']['total_members']} members, {context['team_context']['available_members']} available
- Business impact level: {context['business_context']['impact_level']}
- Employee's leave pattern: {context['user_history']['pattern']}

Make your decision based on holistic analysis, not just rigid rules.
Respond with ONLY valid JSON, no other text.
"""

            logger.info("🧠 Sending request to AI reasoning engine...")

            # FIXED: Remove response_format parameter and use gpt-3.5-turbo for better compatibility
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # Changed from gpt-4 for better compatibility
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent reasoning
                max_tokens=1000   # Ensure we get a complete response
            )

            ai_response = response.choices[0].message.content.strip()
            logger.info(f"🤖 Raw AI Response: {ai_response}")

            # FIXED: Better JSON parsing with fallback
            ai_decision = self._parse_ai_response(ai_response)
            logger.info(f"🤖 Parsed AI Decision: {ai_decision}")

            return ai_decision

        except Exception as e:
            logger.error(f"AI reasoning engine failed: {e}")
            return {
                'status': 'Escalate',
                'reason': 'AI analysis unavailable, requiring human review',
                'confidence': 0.0,
                'escalate': True,
                'agent_reasoning': f"Technical error: {str(e)}"
            }

    def _parse_ai_response(self, response_text: str) -> Dict:
        """
        Parse AI response with robust error handling
        """
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                ai_decision = json.loads(json_str)

                # Validate required fields
                required_fields = ['status', 'reason', 'confidence']
                for field in required_fields:
                    if field not in ai_decision:
                        ai_decision[field] = self._get_default_value(field)

                # Ensure status is valid
                if ai_decision['status'] not in ['Approved', 'Denied', 'Escalate']:
                    ai_decision['status'] = 'Escalate'

                # Ensure confidence is a number between 0 and 1
                try:
                    ai_decision['confidence'] = max(
                        0.0, min(1.0, float(ai_decision['confidence'])))
                except (ValueError, TypeError):
                    ai_decision['confidence'] = 0.5

                return ai_decision
            else:
                raise ValueError("No JSON found in response")

        except Exception as e:
            logger.error(f"Failed to parse AI response: {e}")
            # Fallback: Create decision based on simple rules
            return self._create_fallback_decision(response_text)

    def _get_default_value(self, field: str):
        """Get default values for missing fields"""
        defaults = {
            'status': 'Escalate',
            'reason': 'Unable to determine reason from AI response',
            'confidence': 0.5,
            'escalate': True,
            'need_more_info': False,
            'info_needed': [],
            'agent_reasoning': 'AI response parsing failed',
            'precedent_used': 'None',
            'exception_made': 'None',
            'learning_notes': 'Response parsing issue'
        }
        return defaults.get(field, 'Unknown')

    def _create_fallback_decision(self, response_text: str) -> Dict:
        """
        Create a fallback decision when AI response parsing fails
        """
        # Simple rule-based fallback
        decision = {
            'status': 'Escalate',
            'reason': 'AI response could not be parsed, defaulting to human review',
            'confidence': 0.3,
            'escalate': True,
            'agent_reasoning': f'Fallback decision due to parsing error. Raw response: {response_text[:100]}...',
            'need_more_info': False,
            'info_needed': [],
            'precedent_used': 'None',
            'exception_made': 'System fallback',
            'learning_notes': 'Need to improve AI response parsing'
        }

        return decision

    def _get_user_leave_history(self, user_id: int) -> Dict:
        """Analyze user's leave history patterns"""
        try:
            # In a real system, this would query the database
            # For demo, we'll simulate some patterns
            recent_leaves = Leave.query.filter_by(user_id=user_id).order_by(
                Leave.start_date.desc()).limit(5).all()

            if recent_leaves:
                total_days = sum(
                    [(leave.end_date - leave.start_date).days + 1 for leave in recent_leaves])
                sick_count = sum(
                    [1 for leave in recent_leaves if 'sick' in leave.reason.lower()])

                return {
                    'total_recent_days': total_days,
                    'recent_requests': len(recent_leaves),
                    'sick_leave_frequency': sick_count,
                    'pattern': 'normal' if total_days < 20 else 'high_usage',
                    'last_leave_date': recent_leaves[0].end_date.strftime('%Y-%m-%d') if recent_leaves else None
                }
            else:
                return {
                    'total_recent_days': 0,
                    'recent_requests': 0,
                    'sick_leave_frequency': 0,
                    'pattern': 'new_employee',
                    'last_leave_date': None
                }
        except Exception as e:
            logger.error(f"Error getting user history: {e}")
            return {'pattern': 'unknown', 'error': str(e)}

    def _analyze_team_workload(self, team_members: List, start_date: str) -> Dict:
        """Analyze team workload and availability"""
        # Simulate team workload analysis
        # In real system, this would check calendars, project deadlines, etc.

        total_members = len(team_members)
        # Simulate some members might be unavailable
        unavailable_count = random.randint(0, max(1, total_members // 3))
        available_count = total_members - unavailable_count

        workload_level = 'high' if available_count <= 2 else 'medium' if available_count <= 4 else 'low'

        return {
            'total_count': total_members,
            'available_count': available_count,
            'unavailable_count': unavailable_count,
            'workload_level': workload_level
        }

    def _assess_business_impact(self, start_date: str, end_date: str) -> Dict:
        """Assess business impact of the leave period"""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        # Check for month/quarter end
        is_month_end = start.day > 25 or end.day > 25
        is_quarter_end = start.month in [3, 6, 9, 12] and start.day > 20

        # Simulate project deadlines (in real system, this would check actual data)
        has_project_deadline = random.random() < 0.3  # 30% chance of project deadline

        impact_factors = []
        if is_month_end:
            impact_factors.append('month_end_reporting')
        if is_quarter_end:
            impact_factors.append('quarter_end_activities')
        if has_project_deadline:
            impact_factors.append('project_deadline')

        impact_level = 'high' if len(
            impact_factors) > 1 else 'medium' if impact_factors else 'low'

        return {
            'impact_level': impact_level,
            'impact_factors': impact_factors,
            'business_critical': is_quarter_end or (has_project_deadline and is_month_end)
        }

    def _find_similar_past_decisions(self, leave_request: LeaveRequest) -> List[Dict]:
        """Find similar past decisions for learning"""
        # In real system, this would search the database for similar cases
        # For demo, we'll simulate some similar cases

        similar_cases = [
            {
                'duration': 7,
                'reason': 'sick',
                'decision': 'Approved',
                'rationale': 'Extended illness with medical documentation'
            },
            {
                'duration': 12,
                'reason': 'sick',
                'decision': 'Escalate',
                'rationale': 'Required management review for extended sick leave'
            }
        ]

        # Filter by similar duration and reason
        relevant_cases = [
            case for case in similar_cases
            if abs(case['duration'] - leave_request.duration_days) <= 5
            and case['reason'].lower() in leave_request.reason.lower()
        ]

        return relevant_cases[:3]  # Return top 3 similar cases

    def _gather_additional_information(self, info_needed: List[str]) -> Dict:
        """Gather additional information as requested by AI"""
        additional_info = {}

        for info_type in info_needed:
            if info_type == 'medical_documentation':
                # In real system, this would check if medical docs are provided
                additional_info['has_medical_docs'] = random.choice([
                                                                    True, False])
            elif info_type == 'manager_input':
                # In real system, this would get manager's input
                additional_info['manager_recommendation'] = random.choice(
                    ['approve', 'deny', 'escalate'])
            elif info_type == 'workload_details':
                # In real system, this would get detailed workload info
                additional_info['detailed_workload'] = {
                    'critical_tasks': random.randint(0, 5)}

        return additional_info

    def _update_agent_memory(self, leave_request: LeaveRequest, decision: Dict, context: Dict):
        """Update agent's memory for future learning"""
        decision_record = {
            'timestamp': datetime.now(),
            'request': leave_request,
            'decision': decision,
            'context': context,
            'outcome': 'pending'  # Would be updated later with actual outcome
        }

        self.decision_memory.append(decision_record)

        # Keep only recent decisions in memory (last 100)
        if len(self.decision_memory) > 100:
            self.decision_memory = self.decision_memory[-100:]

        logger.info(
            f"📚 Updated agent memory. Total decisions: {len(self.decision_memory)}")


def process_leave_with_agentic_ai(leave_request_form, team_members, additional_context=None):
    """Process leave request using truly agentic AI system"""
    try:
        agent = AgenticLeaveSystem()

        # Convert to structured format
        leave_req = LeaveRequest(
            reason=leave_request_form['reason'],
            start_date=leave_request_form['start_date'],
            end_date=leave_request_form['end_date'],
            user_id=current_user.id
        )

        # Make intelligent decision using AI reasoning
        decision = agent.make_intelligent_decision(
            leave_req, team_members, additional_context)

        return decision

    except Exception as e:
        logger.exception("Error in agentic AI processing:")
        return {
            "status": "Error",
            "reason": "System error occurred",
            "confidence": 0.0,
            "escalate": True
        }


# =====================================================
# FLASK ROUTES (Updated to use truly agentic system)
# =====================================================

@app.route('/test-log')
def test_log():
    logger.info("This is a test log message")
    return "Check your log files"


@app.cli.command("init-db")
def init_db():
    """Initialize the database with sample data"""
    with app.app_context():
        db.create_all()
        if not Team.query.first():
            # Create a team first
            team = Team(name="Development Team")
            db.session.add(team)
            db.session.commit()

            # Create admin user
            admin = User(
                username='admin',
                password=generate_password_hash('admin123'),
                role='admin',
                team_id=team.id
            )
            # Create regular user
            user = User(
                username='employee',
                password=generate_password_hash('employee123'),
                role='user',
                team_id=team.id
            )
            db.session.add_all([admin, user])
            db.session.commit()
        print("Database initialized with team and user accounts.")


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid username or password', 'error')
    return render_template('login.html')


@app.route('/dashboard')
@login_required
def dashboard():
    if current_user.role == 'admin':
        leaves = Leave.query.all()
    else:
        leaves = Leave.query.filter_by(user_id=current_user.id).all()
    return render_template('dashboard.html', leaves=leaves)


@app.route('/apply', methods=['GET', 'POST'])
@login_required
def apply_leave():
    if request.method == 'POST':
        try:
            logger.info(f"🚀 Agentic leave request from user {current_user.id}")

            team = Team.query.filter_by(id=current_user.team_id).first()
            if not team:
                logger.error(f"No team found for user {current_user.id}")
                flash('You are not assigned to any team', 'error')
                return redirect(url_for('dashboard'))

            logger.info(
                f"Team context: {team.id} with {len(team.members)} members")

            # Use the truly agentic AI system
            decision = process_leave_with_agentic_ai(
                request.form, team.members)
            logger.info(f"🤖 Agentic AI Decision: {decision}")

            new_leave = Leave(
                user_id=current_user.id,
                start_date=datetime.strptime(
                    request.form['start_date'], '%Y-%m-%d').date(),
                end_date=datetime.strptime(
                    request.form['end_date'], '%Y-%m-%d').date(),
                reason=request.form['reason'],
                status=decision['status'],
                decision_reason=decision.get('reason', 'No reason provided')
            )

            db.session.add(new_leave)
            db.session.commit()
            app.logger.info("✅ Leave saved to database")

            # Enhanced flash messages with AI insights
            confidence_text = f" (AI Confidence: {decision.get('confidence', 0):.0%})"
            reasoning_text = f"\n🤖 AI Reasoning: {decision.get('agent_reasoning', '')}"

            flash_message = f"Leave {decision['status']}! {decision.get('reason', '')}{confidence_text}"
            if decision.get('agent_reasoning'):
                flash_message += f" | {decision['agent_reasoning'][:100]}..."

            flash_type = 'success' if decision['status'] == 'Approved' else 'warning' if decision['status'] == 'Escalate' else 'error'

            flash(flash_message, flash_type)
            return redirect(url_for('dashboard'))

        except ValueError as e:
            app.logger.error(f"Date format error: {str(e)}")
            flash('Invalid date format. Please use YYYY-MM-DD', 'error')
        except Exception as e:
            db.session.rollback()
            app.logger.exception("Error processing leave request:")
            flash('Error processing your leave request. Admin notified.', 'error')

        return redirect(url_for('apply_leave'))

    return render_template('apply_leave.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


# Test route to trigger escalation scenarios
@app.route('/test-escalation')
@login_required
def test_escalation():
    """Test different scenarios that might trigger escalation"""
    test_scenarios = [
        {
            'reason': 'Emergency family situation requiring extended time off',
            'start_date': '2025-06-15',
            'end_date': '2025-07-15',  # 30 days - should escalate
        },
        {
            'reason': 'Sick leave - chronic condition flare-up',
            'start_date': '2025-06-20',
            'end_date': '2025-06-27',  # 8 days sick - might escalate
        },
        {
            'reason': 'Personal leave for mental health',
            'start_date': '2025-06-25',
            'end_date': '2025-07-02',  # 8 days - context dependent
        }
    ]

    results = []
    team = Team.query.filter_by(id=current_user.team_id).first()

    for scenario in test_scenarios:
        decision = process_leave_with_agentic_ai(
            scenario, team.members if team else [])
        results.append({
            'scenario': scenario,
            'decision': decision
        })

    return f"<pre>{json.dumps(results, indent=2, default=str)}</pre>"


if __name__ == '__main__':
    app.run(debug=True)
