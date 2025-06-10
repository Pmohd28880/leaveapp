from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, login_user, login_required, current_user, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User, Leave, Team
from datetime import datetime, timedelta
import os
import logging
from logging.handlers import RotatingFileHandler
import json
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import List, Dict, Optional
import random
import asyncio
from concurrent.futures import ThreadPoolExecutor

# OpenAI Agents SDK imports
from agents import Agent, Runner, function_tool
from pydantic import BaseModel, Field
import nest_asyncio

# Allow nested event loops (needed for Flask + async)
nest_asyncio.apply()

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = "IAmhere"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///leave.db'
db.init_app(app)

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
# PYDANTIC MODELS FOR STRUCTURED OUTPUTS
# =====================================================


class LeaveDecision(BaseModel):
    """Structured output for leave decisions"""
    status: str = Field(
        description="Decision status: Approved, Denied, or Escalate")
    reason: str = Field(description="Detailed explanation for the decision")
    confidence: float = Field(
        description="Confidence level between 0.0 and 1.0", ge=0.0, le=1.0)
    escalate: bool = Field(description="Whether this requires human review")
    agent_reasoning: str = Field(description="Step-by-step reasoning process")
    business_impact: str = Field(description="Assessment of business impact")
    employee_considerations: str = Field(
        description="Employee wellbeing factors considered")
    precedent_used: Optional[str] = Field(
        description="Similar cases referenced", default=None)
    recommended_actions: List[str] = Field(
        description="Any follow-up actions needed", default_factory=list)


class ContextAnalysis(BaseModel):
    """Analysis of leave request context"""
    user_pattern: str = Field(description="Employee's leave usage pattern")
    team_impact: str = Field(description="Impact on team operations")
    business_timing: str = Field(
        description="Business calendar considerations")
    risk_assessment: str = Field(
        description="Overall risk level: low, medium, high")
    recommendations: List[str] = Field(
        description="Recommendations for decision", default_factory=list)

# =====================================================
# DATACLASS FOR LEAVE REQUESTS
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

# =====================================================
# AGENT TOOLS (Functions agents can call)
# =====================================================


@function_tool
def get_user_leave_history(user_id: int) -> Dict:
    """Retrieve and analyze user's leave history patterns"""
    try:
        logger.info(f"🔍 Analyzing leave history for user {user_id}")

        # Get recent leave requests
        recent_leaves = Leave.query.filter_by(user_id=user_id).order_by(
            Leave.start_date.desc()).limit(10).all()

        if recent_leaves:
            total_days = sum(
                [(leave.end_date - leave.start_date).days + 1 for leave in recent_leaves])
            sick_count = len(
                [leave for leave in recent_leaves if 'sick' in leave.reason.lower()])
            approved_count = len(
                [leave for leave in recent_leaves if leave.status == 'Approved'])

            # Calculate patterns
            avg_days_per_request = total_days / \
                len(recent_leaves) if recent_leaves else 0
            approval_rate = (approved_count / len(recent_leaves)
                             ) * 100 if recent_leaves else 0

            # Determine pattern classification
            if total_days > 30:
                pattern = "high_usage"
            elif sick_count > len(recent_leaves) * 0.7:
                pattern = "frequent_sick_leave"
            elif avg_days_per_request > 7:
                pattern = "long_duration_requests"
            else:
                pattern = "normal"

            return {
                'total_recent_days': total_days,
                'total_requests': len(recent_leaves),
                'sick_leave_count': sick_count,
                'approval_rate': approval_rate,
                'average_duration': avg_days_per_request,
                'pattern_classification': pattern,
                'last_leave_date': recent_leaves[0].end_date.strftime('%Y-%m-%d'),
                'recent_reasons': [leave.reason for leave in recent_leaves[:3]]
            }
        else:
            return {
                'total_recent_days': 0,
                'total_requests': 0,
                'pattern_classification': 'new_employee',
                'message': 'No previous leave history found'
            }

    except Exception as e:
        logger.error(f"Error analyzing user history: {e}")
        return {'error': str(e), 'pattern_classification': 'unknown'}


@function_tool
def analyze_team_availability(team_id: int, start_date: str, end_date: str) -> Dict:
    """Analyze team availability during requested leave period"""
    try:
        logger.info(f"👥 Analyzing team availability for team {team_id}")

        # Get team members
        team = Team.query.get(team_id)
        if not team:
            return {'error': 'Team not found'}

        total_members = len(team.members)

        # Check for overlapping leaves during the requested period
        start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()

        overlapping_leaves = Leave.query.join(User).filter(
            User.team_id == team_id,
            Leave.status == 'Approved',
            Leave.start_date <= end_dt,
            Leave.end_date >= start_dt
        ).all()

        members_on_leave = len(
            set([leave.user_id for leave in overlapping_leaves]))
        available_members = total_members - members_on_leave

        # Calculate impact
        if available_members <= 1:
            impact_level = "critical"
        elif available_members <= total_members * 0.5:
            impact_level = "high"
        elif available_members <= total_members * 0.7:
            impact_level = "medium"
        else:
            impact_level = "low"

        return {
            'total_team_members': total_members,
            'members_on_leave': members_on_leave,
            'available_members': available_members,
            'availability_percentage': (available_members / total_members) * 100,
            'impact_level': impact_level,
            'overlapping_leaves': len(overlapping_leaves),
            'team_name': team.name
        }

    except Exception as e:
        logger.error(f"Error analyzing team availability: {e}")
        return {'error': str(e), 'impact_level': 'unknown'}


@function_tool
def assess_business_calendar_impact(start_date: str, end_date: str) -> Dict:
    """Assess impact based on business calendar and critical periods"""
    try:
        logger.info(
            f"📅 Assessing business calendar impact for {start_date} to {end_date}")

        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        critical_periods = []
        impact_factors = []

        # Check for month-end (last 5 days of month)
        if start_dt.day >= 26 or end_dt.day >= 26:
            critical_periods.append("month_end_reporting")
            impact_factors.append(
                "Financial reporting and month-end processes")

        # Check for quarter-end
        quarter_end_months = [3, 6, 9, 12]
        if (start_dt.month in quarter_end_months and start_dt.day >= 25) or \
           (end_dt.month in quarter_end_months and end_dt.day >= 25):
            critical_periods.append("quarter_end")
            impact_factors.append(
                "Quarter-end financial activities and reporting")

        # Check for year-end
        if (start_dt.month == 12 and start_dt.day >= 20) or \
           (end_dt.month == 1 and end_dt.day <= 10):
            critical_periods.append("year_end")
            impact_factors.append("Year-end closing and planning activities")

        # Simulate project deadlines (in real system, integrate with project management)
        project_risk = random.choice([True, False])  # 50% chance
        if project_risk:
            critical_periods.append("project_deadlines")
            impact_factors.append("Potential project milestone deadlines")

        # Determine overall business impact
        if len(critical_periods) >= 2:
            business_impact = "high"
        elif len(critical_periods) == 1:
            business_impact = "medium"
        else:
            business_impact = "low"

        return {
            'business_impact_level': business_impact,
            'critical_periods': critical_periods,
            'impact_factors': impact_factors,
            'recommendation': "Require additional approval" if business_impact == "high" else "Standard process",
            'start_date': start_date,
            'end_date': end_date
        }

    except Exception as e:
        logger.error(f"Error assessing business impact: {e}")
        return {'error': str(e), 'business_impact_level': 'unknown'}


@function_tool
def get_similar_leave_decisions(reason: str, duration_days: int, user_id: int) -> Dict:
    """Find similar past leave decisions for precedent analysis"""
    try:
        logger.info(
            f"🔍 Finding similar decisions for {reason}, {duration_days} days")

        # Find leaves with similar characteristics
        similar_leaves = Leave.query.filter(
            Leave.user_id != user_id,  # Different users
            Leave.reason.contains(reason.split()[0])  # Similar reason
        ).limit(5).all()

        if not similar_leaves:
            return {'message': 'No similar cases found', 'precedents': []}

        precedents = []
        for leave in similar_leaves:
            leave_duration = (leave.end_date - leave.start_date).days + 1
            precedents.append({
                'duration': leave_duration,
                'reason': leave.reason,
                'status': leave.status,
                'decision_reason': leave.decision_reason,
                'similarity_score': max(0, 100 - abs(leave_duration - duration_days) * 5)
            })

        # Sort by similarity
        precedents.sort(key=lambda x: x['similarity_score'], reverse=True)

        approval_rate = len(
            [p for p in precedents if p['status'] == 'Approved']) / len(precedents) * 100

        return {
            'similar_cases_found': len(precedents),
            'precedents': precedents[:3],  # Top 3 most similar
            'historical_approval_rate': approval_rate,
            'recommendation': 'Favor approval' if approval_rate > 70 else 'Standard review' if approval_rate > 40 else 'Careful review'
        }

    except Exception as e:
        logger.error(f"Error finding similar decisions: {e}")
        return {'error': str(e)}

# =====================================================
# SPECIALIZED AGENTS
# =====================================================


# Context Analysis Agent
context_agent = Agent(
    name="Leave Context Analyzer",
    instructions="""
    You are a specialized agent for analyzing the context around leave requests.

    Your role is to:
    1. Use available tools to gather comprehensive information
    2. Analyze patterns and identify potential concerns
    3. Provide structured assessment of the situation

    Focus on:
    - Employee leave patterns and history
    - Team impact and availability
    - Business timing considerations
    - Risk factors that need attention

    Be thorough but efficient in your analysis.
    """,
    tools=[get_user_leave_history, analyze_team_availability,
           assess_business_calendar_impact, get_similar_leave_decisions],
    output_type=ContextAnalysis
)

# Decision Making Agent
decision_agent = Agent(
    name="Leave Decision Maker",
    instructions="""
    You are the primary decision-making agent for employee leave requests.

    Core principles:
    1. Employee wellbeing is paramount
    2. Business continuity must be maintained
    3. Fairness and consistency in decisions
    4. Transparency in reasoning
    5. Escalate when human judgment is needed

    Decision guidelines:
    - Approve when impact is manageable and request is reasonable
    - Deny only when business impact is severe and no alternatives exist
    - Escalate for complex situations, policy exceptions, or high-risk scenarios

    Consider:
    - Employee's history and circumstances
    - Team capacity and availability
    - Business timing and critical periods
    - Precedent from similar cases
    - Company values of flexibility and employee support

    Always provide clear, empathetic reasoning for your decisions.
    """,
    tools=[get_user_leave_history, analyze_team_availability,
           assess_business_calendar_impact, get_similar_leave_decisions],
    output_type=LeaveDecision
)

# Escalation Agent
escalation_agent = Agent(
    name="Leave Escalation Specialist",
    instructions="""
    You handle complex and sensitive leave requests that require special consideration.

    Your expertise covers:
    - Extended leave requests (>14 days)
    - Medical and family emergencies
    - Unusual circumstances
    - Policy exceptions
    - High business impact situations

    Approach:
    - Gather comprehensive context
    - Consider all stakeholders
    - Identify accommodation options
    - Provide detailed analysis for human reviewers
    - Suggest alternative solutions when possible

    Remember: Your goal is to find solutions that work for both employee and business.
    """,
    tools=[get_user_leave_history, analyze_team_availability,
           assess_business_calendar_impact, get_similar_leave_decisions],
    output_type=LeaveDecision
)

# Triage Agent (Entry point)
triage_agent = Agent(
    name="Leave Request Triage",
    instructions="""
    You are the first point of contact for all leave requests. Your job is to:

    1. Quickly assess the complexity and sensitivity of the request
    2. Route to the appropriate specialist agent
    3. Ensure efficient processing while maintaining quality

    Routing guidelines:

    → Standard Decision Agent for:
    - Routine vacation requests (≤14 days)
    - Standard sick leave (≤5 days)
    - Personal leave with clear reasons
    - Low team/business impact situations

    → Escalation Specialist for:
    - Extended leave requests (>14 days)
    - Medical emergencies or serious illness
    - Family crisis situations
    - Requests during critical business periods
    - Employees with concerning leave patterns
    - Policy exception requests
    - High business impact scenarios

    Always explain your routing decision briefly.
    """,
    handoffs=[decision_agent, escalation_agent]
)

# =====================================================
# AGENTS SDK INTEGRATION CLASS
# =====================================================


class AgenticLeaveSystemSDK:
    def __init__(self):
        self.triage_agent = triage_agent
        self.context_agent = context_agent
        self.executor = ThreadPoolExecutor(max_workers=3)

    def make_intelligent_decision(self, leave_request: LeaveRequest, team_members: List, additional_context: Dict = None) -> Dict:
        """
        Process leave request using OpenAI Agents SDK
        """
        try:
            logger.info(
                f"🤖 Starting Agents SDK analysis for user {leave_request.user_id}")

            # Run the async processing in the thread executor
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                result = loop.run_until_complete(
                    self._async_process_request(
                        leave_request, team_members, additional_context)
                )
                return result
            finally:
                loop.close()

        except Exception as e:
            logger.exception("Error in Agents SDK processing:")
            return {
                'status': 'Escalate',
                'reason': 'System error during AI analysis',
                'confidence': 0.0,
                'escalate': True,
                'agent_reasoning': f'SDK Error: {str(e)}',
                'business_impact': 'unknown',
                'employee_considerations': 'system_error'
            }

    async def _async_process_request(self, leave_request: LeaveRequest, team_members: List, additional_context: Dict = None) -> Dict:
        """Async processing of leave request"""

        # Get user's team for context
        user = User.query.get(leave_request.user_id)
        team_id = user.team_id if user else None

        # Format comprehensive input for the triage agent
        agent_input = f"""
        NEW LEAVE REQUEST FOR ANALYSIS:

        Employee Details:
        - User ID: {leave_request.user_id}
        - Team ID: {team_id}

        Leave Request:
        - Reason: {leave_request.reason}
        - Duration: {leave_request.duration_days} days
        - Start Date: {leave_request.start_date}
        - End Date: {leave_request.end_date}

        Team Context:
        - Team Size: {len(team_members)} members

        Additional Context: {json.dumps(additional_context or {}, indent=2)}

        Please analyze this request thoroughly and make an appropriate decision.
        Use all available tools to gather context before deciding.
        """

        logger.info("🚀 Sending request to Agents SDK triage system...")

        # Run the agent
        result = await Runner.run(self.triage_agent, agent_input)

        logger.info(f"🤖 Agents SDK completed processing")
        logger.info(f"📊 Result type: {type(result.final_output)}")

        # Convert the structured output to dict
        if isinstance(result.final_output, LeaveDecision):
            decision = result.final_output
            return {
                'status': decision.status,
                'reason': decision.reason,
                'confidence': decision.confidence,
                'escalate': decision.escalate,
                'agent_reasoning': decision.agent_reasoning,
                'business_impact': decision.business_impact,
                'employee_considerations': decision.employee_considerations,
                'precedent_used': decision.precedent_used,
                'recommended_actions': decision.recommended_actions
            }
        else:
            # Handle unexpected output format
            logger.warning(
                f"Unexpected output type: {type(result.final_output)}")
            return {
                'status': 'Escalate',
                'reason': 'Unexpected AI response format',
                'confidence': 0.5,
                'escalate': True,
                'agent_reasoning': f'Output: {str(result.final_output)}',
                'business_impact': 'unknown',
                'employee_considerations': 'format_error'
            }

# =====================================================
# FLASK ROUTES
# =====================================================


@app.route('/test-log')
def test_log():
    logger.info("This is a test log message from Agents SDK version")
    return "Check your log files - Agents SDK version active"


@app.cli.command("init-db")
def init_db():
    """Initialize the database with sample data"""
    with app.app_context():
        db.create_all()
        if not Team.query.first():
            # Create teams
            dev_team = Team(name="Development Team")
            hr_team = Team(name="HR Team")
            db.session.add_all([dev_team, hr_team])
            db.session.commit()

            # Create users
            admin = User(
                username='admin',
                password=generate_password_hash('admin123'),
                role='admin',
                team_id=dev_team.id
            )
            employee1 = User(
                username='employee',
                password=generate_password_hash('employee123'),
                role='user',
                team_id=dev_team.id
            )
            employee2 = User(
                username='john',
                password=generate_password_hash('john123'),
                role='user',
                team_id=dev_team.id
            )
            hr_user = User(
                username='hr_manager',
                password=generate_password_hash('hr123'),
                role='admin',
                team_id=hr_team.id
            )

            db.session.add_all([admin, employee1, employee2, hr_user])
            db.session.commit()

            # Add some sample leave history
            sample_leave = Leave(
                user_id=employee1.id,
                start_date=datetime(2024, 5, 1).date(),
                end_date=datetime(2024, 5, 3).date(),
                reason="Sick leave - flu",
                status="Approved",
                decision_reason="Standard sick leave approval"
            )
            db.session.add(sample_leave)
            db.session.commit()

        print("Database initialized with teams, users, and sample data for Agents SDK version.")


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            flash(
                f'Welcome back! Now using AI Agents SDK for intelligent leave decisions.', 'success')
            return redirect(url_for('dashboard'))
        flash('Invalid username or password', 'error')
    return render_template('login.html')


@app.route('/dashboard')
@login_required
def dashboard():
    if current_user.role == 'admin':
        leaves = Leave.query.order_by(Leave.created_at.desc()).all()
    else:
        leaves = Leave.query.filter_by(user_id=current_user.id).order_by(
            Leave.created_at.desc()).all()
    return render_template('dashboard.html', leaves=leaves)


@app.route('/apply', methods=['GET', 'POST'])
@login_required
def apply_leave():
    if request.method == 'POST':
        try:
            logger.info(
                f"🚀 New Agents SDK leave request from user {current_user.id}")

            # Validate team assignment
            if not current_user.team_id:
                flash('You are not assigned to any team. Please contact HR.', 'error')
                return redirect(url_for('dashboard'))

            team = Team.query.get(current_user.team_id)
            if not team:
                flash('Your team information is invalid. Please contact HR.', 'error')
                return redirect(url_for('dashboard'))

            logger.info(
                f"Team context: {team.name} ({team.id}) with {len(team.members)} members")

            # Create leave request object
            leave_request = LeaveRequest(
                reason=request.form['reason'],
                start_date=request.form['start_date'],
                end_date=request.form['end_date'],
                user_id=current_user.id
            )

            # Process with Agents SDK
            agent_system = AgenticLeaveSystemSDK()
            decision = agent_system.make_intelligent_decision(
                leave_request,
                team.members,
                {
                    'request_source': 'web_form',
                    'user_role': current_user.role,
                    'team_name': team.name
                }
            )

            logger.info(f"🤖 Agents SDK Decision: {decision}")

            # Save to database
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
            logger.info("✅ Leave saved to database")

            # Enhanced flash messages with Agents SDK insights
            confidence_text = f" (AI Confidence: {decision.get('confidence', 0):.0%})"

            flash_message = f"🤖 Leave {decision['status']}! {decision.get('reason', '')}{confidence_text}"

            # Add business impact info if available
            if decision.get('business_impact') and decision['business_impact'] != 'unknown':
                flash_message += f" | Business Impact: {decision['business_impact']}"

            # Add employee considerations
            if decision.get('employee_considerations') and decision['employee_considerations'] != 'system_error':
                flash_message += f" | Employee factors considered: {decision['employee_considerations']}"

            flash_type = 'success' if decision['status'] == 'Approved' else 'warning' if decision['status'] == 'Escalate' else 'info'
            flash(flash_message, flash_type)

            # Show reasoning in a separate message for transparency
            if decision.get('agent_reasoning'):
                reasoning_msg = f"🧠 AI Reasoning: {decision['agent_reasoning'][:200]}..."
                flash(reasoning_msg, 'info')

            return redirect(url_for('dashboard'))

        except ValueError as e:
            logger.error(f"Date format error: {str(e)}")
            flash('Invalid date format. Please use the date picker.', 'error')
        except Exception as e:
            db.session.rollback()
            logger.exception("Error processing leave request with Agents SDK:")
            flash('Error processing your leave request. The issue has been logged and HR has been notified.', 'error')

        return redirect(url_for('apply_leave'))

    return render_template('apply_leave.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('login'))

# Enhanced test routes for Agents SDK


@app.route('/test-agents-sdk')
@login_required
def test_agents_sdk():
    """Test the Agents SDK with various scenarios"""

    test_scenarios = [
        {
            'name': 'Standard Vacation Request',
            'reason': 'Annual vacation with family',
            'start_date': '2025-07-01',
            'end_date': '2025-07-05',  # 5 days
        },
        {
            'name': 'Extended Sick Leave',
            'reason': 'Medical procedure and recovery time needed',
            'start_date': '2025-06-20',
            'end_date': '2025-07-10',  # 21 days - should escalate
        },
        {
            'name': 'Emergency Family Situation',
            'reason': 'Family emergency requiring immediate attention',
            'start_date': '2025-06-15',
            'end_date': '2025-06-22',  # 8 days
        },
        {
            'name': 'Mental Health Leave',
            'reason': 'Personal mental health and wellbeing break',
            'start_date': '2025-06-25',
            'end_date': '2025-07-02',  # 8 days
        }
    ]

    results = []
    team = Team.query.get(
        current_user.team_id) if current_user.team_id else None

    if not team:
        return "Error: User not assigned to a team", 400

    agent_system = AgenticLeaveSystemSDK()

    for scenario in test_scenarios:
        try:
            leave_req = LeaveRequest(
                reason=scenario['reason'],
                start_date=scenario['start_date'],
                end_date=scenario['end_date'],
                user_id=current_user.id
            )

            decision = agent_system.make_intelligent_decision(
                leave_req,
                team.members,
                {'test_scenario': scenario['name']}
            )

            results.append({
                'scenario': scenario['name'],
                'request': scenario,
                'decision': decision
            })

        except Exception as e:
            results.append({
                'scenario': scenario['name'],
                'request': scenario,
                'error': str(e)
            })

    # Format results as HTML for easy viewing
    html_output = "<h2>🤖 Agents SDK Test Results</h2>"
    for result in results:
        html_output += f"<h3>📋 {result['scenario']}</h3>"
        html_output += f"<pre>{json.dumps(result, indent=2, default=str)}</pre><hr>"

    return html_output


@app.route('/test-escalation')
@login_required
def test_escalation():
    """Test different scenarios that might trigger escalation"""

    # Check if user has admin privileges (optional - you might want to restrict this)
    if current_user.role != 'admin':
        flash('Access denied. Admin privileges required for testing escalation scenarios.', 'error')
        return redirect(url_for('dashboard'))

    test_scenarios = [
        {
            'name': 'Extended Family Emergency',
            'reason': 'Emergency family situation requiring extended time off',
            'start_date': '2025-06-15',
            'end_date': '2025-07-15',  # 30 days - should escalate
            'expected_outcome': 'Escalate - Extended duration'
        },
        {
            'name': 'Chronic Condition Sick Leave',
            'reason': 'Sick leave - chronic condition flare-up',
            'start_date': '2025-06-20',
            'end_date': '2025-06-27',  # 8 days sick - might escalate
            'expected_outcome': 'Escalate or Approve - Medical consideration'
        },
        {
            'name': 'Mental Health Personal Leave',
            'reason': 'Personal leave for mental health and wellbeing',
            'start_date': '2025-06-25',
            'end_date': '2025-07-02',  # 8 days - context dependent
            'expected_outcome': 'Context dependent - Employee wellbeing priority'
        },
        {
            'name': 'Medical Procedure Extended Leave',
            'reason': 'Scheduled medical procedure requiring recovery time',
            'start_date': '2025-07-01',
            'end_date': '2025-07-28',  # 28 days - should escalate
            'expected_outcome': 'Escalate - Extended medical leave'
        },
        {
            'name': 'Bereavement Leave Extended',
            'reason': 'Bereavement leave for close family member - additional arrangements needed',
            'start_date': '2025-06-18',
            'end_date': '2025-06-30',  # 13 days - might escalate
            'expected_outcome': 'Escalate - Extended bereavement consideration'
        },
        {
            'name': 'Quarter-End Critical Period Request',
            'reason': 'Personal vacation during busy period',
            'start_date': '2025-09-25',  # Quarter end period
            'end_date': '2025-09-30',   # 6 days during critical period
            'expected_outcome': 'Escalate - Business critical timing'
        }
    ]

    results = []

    # Get team information
    team = Team.query.filter_by(id=current_user.team_id).first()
    if not team:
        return "Error: User not assigned to a team. Cannot run escalation tests.", 400

    # Initialize the Agents SDK system
    agent_system = AgenticLeaveSystemSDK()

    logger.info(
        f"🧪 Running escalation tests for {len(test_scenarios)} scenarios")

    for scenario in test_scenarios:
        try:
            logger.info(f"Testing scenario: {scenario['name']}")

            # Create leave request object
            leave_request = LeaveRequest(
                reason=scenario['reason'],
                start_date=scenario['start_date'],
                end_date=scenario['end_date'],
                user_id=current_user.id
            )

            # Process with Agents SDK
            decision = agent_system.make_intelligent_decision(
                leave_request,
                team.members,
                {
                    'test_scenario': scenario['name'],
                    'escalation_test': True,
                    'expected_outcome': scenario['expected_outcome']
                }
            )

            # Analyze if escalation was triggered as expected
            escalation_triggered = decision.get(
                'escalate', False) or decision.get('status') == 'Escalate'

            results.append({
                'scenario_name': scenario['name'],
                'request_details': {
                    'reason': scenario['reason'],
                    'duration_days': leave_request.duration_days,
                    'start_date': scenario['start_date'],
                    'end_date': scenario['end_date']
                },
                'expected_outcome': scenario['expected_outcome'],
                'actual_decision': {
                    'status': decision.get('status'),
                    'escalate': decision.get('escalate', False),
                    'confidence': decision.get('confidence', 0),
                    'reason': decision.get('reason', 'No reason provided'),
                    'business_impact': decision.get('business_impact', 'unknown'),
                    'employee_considerations': decision.get('employee_considerations', 'none')
                },
                'escalation_triggered': escalation_triggered,
                'agent_reasoning': decision.get('agent_reasoning', 'No reasoning provided'),
                'test_result': 'PASS' if escalation_triggered else 'REVIEW_NEEDED'
            })

        except Exception as e:
            logger.error(
                f"Error testing scenario {scenario['name']}: {str(e)}")
            results.append({
                'scenario_name': scenario['name'],
                'request_details': scenario,
                'error': str(e),
                'test_result': 'ERROR'
            })

    # Generate summary statistics
    total_tests = len(results)
    escalations_triggered = len(
        [r for r in results if r.get('escalation_triggered', False)])
    errors = len([r for r in results if 'error' in r])

    summary = {
        'total_scenarios_tested': total_tests,
        'escalations_triggered': escalations_triggered,
        'escalation_rate': f"{(escalations_triggered/total_tests)*100:.1f}%" if total_tests > 0 else "0%",
        'errors_encountered': errors,
        'test_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'tester_user_id': current_user.id,
        'team_context': team.name if team else 'No team'
    }

    # Format results as HTML for easy viewing
    html_output = f"""
    <div style="font-family: Arial, sans-serif; margin: 20px;">
        <h2>🚨 Escalation Testing Results</h2>
        
        <div style="background: #f0f8ff; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
            <h3>📊 Test Summary</h3>
            <ul>
                <li><strong>Total Scenarios:</strong> {summary['total_scenarios_tested']}</li>
                <li><strong>Escalations Triggered:</strong> {summary['escalations_triggered']}</li>
                <li><strong>Escalation Rate:</strong> {summary['escalation_rate']}</li>
                <li><strong>Errors:</strong> {summary['errors_encountered']}</li>
                <li><strong>Test Time:</strong> {summary['test_timestamp']}</li>
                <li><strong>Team Context:</strong> {summary['team_context']}</li>
            </ul>
        </div>
    """

    # Add detailed results for each scenario
    for i, result in enumerate(results, 1):
        status_color = '#28a745' if result.get(
            'escalation_triggered') else '#ffc107'
        if 'error' in result:
            status_color = '#dc3545'

        html_output += f"""
        <div style="border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px;">
            <h3 style="color: {status_color};">#{i}. {result['scenario_name']} 
                <span style="font-size: 0.8em; background: {status_color}; color: white; padding: 2px 8px; border-radius: 3px;">
                    {result.get('test_result', 'UNKNOWN')}
                </span>
            </h3>
            
            <div style="display: flex; gap: 20px;">
                <div style="flex: 1;">
                    <h4>📋 Request Details:</h4>
                    <ul>
                        <li><strong>Reason:</strong> {result.get('request_details', {}).get('reason', 'N/A')}</li>
                        <li><strong>Duration:</strong> {result.get('request_details', {}).get('duration_days', 'N/A')} days</li>
                        <li><strong>Dates:</strong> {result.get('request_details', {}).get('start_date', 'N/A')} to {result.get('request_details', {}).get('end_date', 'N/A')}</li>
                    </ul>
                </div>
                
                <div style="flex: 1;">
                    <h4>🤖 AI Decision:</h4>
                    <ul>
                        <li><strong>Status:</strong> {result.get('actual_decision', {}).get('status', 'N/A')}</li>
                        <li><strong>Escalate:</strong> {result.get('actual_decision', {}).get('escalate', False)}</li>
                        <li><strong>Confidence:</strong> {result.get('actual_decision', {}).get('confidence', 0):.1%}</li>
                        <li><strong>Business Impact:</strong> {result.get('actual_decision', {}).get('business_impact', 'N/A')}</li>
                    </ul>
                </div>
            </div>
            
            <div style="margin-top: 10px;">
                <h4>💭 Decision Reasoning:</h4>
                <p style="background: #f8f9fa; padding: 10px; border-radius: 3px; font-style: italic;">
                    "{result.get('agent_reasoning', 'No reasoning provided')}"
                </p>
            </div>
            
            <div style="margin-top: 10px;">
                <h4>🎯 Expected vs Actual:</h4>
                <p><strong>Expected:</strong> {result.get('expected_outcome', 'N/A')}</p>
                <p><strong>Actual Decision Reason:</strong> {result.get('actual_decision', {}).get('reason', 'N/A')}</p>
            </div>
            
            {f'<div style="color: red; margin-top: 10px;"><strong>❌ Error:</strong> {result["error"]}</div>' if 'error' in result else ''}
        </div>
        """

    html_output += """
        <div style="margin-top: 30px; padding: 15px; background: #e9ecef; border-radius: 5px;">
            <h3>📝 Notes:</h3>
            <ul>
                <li>Escalation scenarios test the AI's ability to identify complex situations requiring human review</li>
                <li>High confidence escalations indicate the AI is correctly identifying edge cases</li>
                <li>Review scenarios marked as "REVIEW_NEEDED" to ensure appropriate escalation triggers</li>
                <li>Extended leave requests (>14 days) and medical situations should typically escalate</li>
            </ul>
        </div>
    </div>
    """

    # Log the test results
    logger.info(f"🧪 Escalation test completed: {summary}")

    return html_output


@app.route('/agent-analytics')
@login_required
def agent_analytics():
    """Show analytics and insights from the AI agents"""
    if current_user.role != 'admin':
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('dashboard'))

    # Get recent leaves with decisions
    recent_leaves = Leave.query.order_by(
        Leave.created_at.desc()).limit(50).all()

    # Calculate analytics
    analytics = {
        'total_requests': len(recent_leaves),
        'approved_count': len([l for l in recent_leaves if l.status == 'Approved']),
        'denied_count': len([l for l in recent_leaves if l.status == 'Denied']),
        'escalated_count': len([l for l in recent_leaves if l.status == 'Escalate']),
        'pending_count': len([l for l in recent_leaves if l.status == 'Pending']),
    }

    # Calculate rates
    if analytics['total_requests'] > 0:
        analytics['approval_rate'] = (
            analytics['approved_count'] / analytics['total_requests']) * 100
        analytics['escalation_rate'] = (
            analytics['escalated_count'] / analytics['total_requests']) * 100
        analytics['denial_rate'] = (
            analytics['denied_count'] / analytics['total_requests']) * 100
    else:
        analytics['approval_rate'] = 0
        analytics['escalation_rate'] = 0
        analytics['denial_rate'] = 0

    # Analyze leave patterns by reason
    reason_patterns = {}
    for leave in recent_leaves:
        reason_key = leave.reason.lower()
        if 'sick' in reason_key:
            category = 'Sick Leave'
        elif 'vacation' in reason_key or 'holiday' in reason_key:
            category = 'Vacation'
        elif 'personal' in reason_key:
            category = 'Personal'
        elif 'emergency' in reason_key or 'family' in reason_key:
            category = 'Emergency/Family'
        elif 'medical' in reason_key:
            category = 'Medical'
        else:
            category = 'Other'

        if category not in reason_patterns:
            reason_patterns[category] = {
                'count': 0, 'approved': 0, 'denied': 0, 'escalated': 0}

        reason_patterns[category]['count'] += 1
        if leave.status == 'Approved':
            reason_patterns[category]['approved'] += 1
        elif leave.status == 'Denied':
            reason_patterns[category]['denied'] += 1
        elif leave.status == 'Escalate':
            reason_patterns[category]['escalated'] += 1

    # Team-wise analytics
    team_analytics = {}
    teams = Team.query.all()
    for team in teams:
        team_leaves = [l for l in recent_leaves if l.user.team_id == team.id]
        if team_leaves:
            team_analytics[team.name] = {
                'total_requests': len(team_leaves),
                'approved': len([l for l in team_leaves if l.status == 'Approved']),
                'denied': len([l for l in team_leaves if l.status == 'Denied']),
                'escalated': len([l for l in team_leaves if l.status == 'Escalate']),
                'avg_duration': sum([(l.end_date - l.start_date).days + 1 for l in team_leaves]) / len(team_leaves)
            }

    # Duration analysis
    duration_analytics = {
        'short_term': len([l for l in recent_leaves if (l.end_date - l.start_date).days <= 2]),
        'medium_term': len([l for l in recent_leaves if 3 <= (l.end_date - l.start_date).days <= 7]),
        'long_term': len([l for l in recent_leaves if (l.end_date - l.start_date).days > 7]),
    }

    # Recent trends (last 30 days vs previous 30 days)
    thirty_days_ago = datetime.now().date() - timedelta(days=30)
    sixty_days_ago = datetime.now().date() - timedelta(days=60)

    recent_month = [l for l in recent_leaves if l.created_at.date()
                    >= thirty_days_ago]
    previous_month = [l for l in recent_leaves if sixty_days_ago <=
                      l.created_at.date() < thirty_days_ago]

    trends = {
        'current_month_requests': len(recent_month),
        'previous_month_requests': len(previous_month),
        'trend_direction': 'up' if len(recent_month) > len(previous_month) else 'down' if len(recent_month) < len(previous_month) else 'stable'
    }

    if len(previous_month) > 0:
        trends['percentage_change'] = (
            (len(recent_month) - len(previous_month)) / len(previous_month)) * 100
    else:
        trends['percentage_change'] = 0

    # Most common decision reasons
    decision_reasons = {}
    for leave in recent_leaves:
        if leave.decision_reason:
            reason = leave.decision_reason[:50] + "..." if len(
                leave.decision_reason) > 50 else leave.decision_reason
            decision_reasons[reason] = decision_reasons.get(reason, 0) + 1

    # Sort by frequency and get top 5
    top_decision_reasons = sorted(
        decision_reasons.items(), key=lambda x: x[1], reverse=True)[:5]

    return render_template('analytics.html',
                           analytics=analytics,
                           reason_patterns=reason_patterns,
                           team_analytics=team_analytics,
                           duration_analytics=duration_analytics,
                           trends=trends,
                           top_decision_reasons=top_decision_reasons,
                           # Show last 10 for quick view
                           recent_leaves=recent_leaves[:10])

# Additional route for AI agent performance metrics


@app.route('/ai-performance')
@login_required
def ai_performance():
    """Show AI agent performance metrics"""
    if current_user.role != 'admin':
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('dashboard'))

    # Get leaves processed by AI (those with decision_reason containing AI indicators)
    ai_processed_leaves = Leave.query.filter(
        Leave.decision_reason.like('%AI%') |
        Leave.decision_reason.like('%agent%') |
        Leave.decision_reason.like('%confidence%')
    ).order_by(Leave.created_at.desc()).limit(100).all()

    # Performance metrics
    performance_metrics = {
        'total_ai_decisions': len(ai_processed_leaves),
        'avg_processing_time': 'N/A',  # Would need to track processing time
        'escalation_accuracy': 0,  # Would need human feedback to calculate
        'user_satisfaction': 'N/A',  # Would need user ratings
    }

    # AI decision patterns
    ai_decision_patterns = {
        'auto_approved': len([l for l in ai_processed_leaves if l.status == 'Approved']),
        'auto_denied': len([l for l in ai_processed_leaves if l.status == 'Denied']),
        'escalated_to_human': len([l for l in ai_processed_leaves if l.status == 'Escalate']),
    }

    # Confidence score analysis (extract from decision_reason if available)
    confidence_scores = []
    for leave in ai_processed_leaves:
        if 'confidence' in leave.decision_reason.lower():
            # Try to extract confidence percentage
            import re
            match = re.search(r'(\d+)%', leave.decision_reason)
            if match:
                confidence_scores.append(int(match.group(1)))

    avg_confidence = sum(confidence_scores) / \
        len(confidence_scores) if confidence_scores else 0

    return render_template('ai_performance.html',
                           performance_metrics=performance_metrics,
                           ai_decision_patterns=ai_decision_patterns,
                           avg_confidence=avg_confidence,
                           confidence_scores=confidence_scores,
                           ai_processed_leaves=ai_processed_leaves[:20])

# Error handlers


@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    logger.error(f"Internal server error: {error}")
    return render_template('500.html'), 500

# Health check endpoint


@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Test database connection
        db.session.execute('SELECT 1')

        # Test agents SDK (basic initialization)
        agent_system = AgenticLeaveSystemSDK()

        return {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'database': 'connected',
            'agents_sdk': 'initialized',
            'version': 'agents-sdk-v1.0'
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }, 500

# Development utilities


@app.route('/dev/reset-db')
def reset_database():
    """Reset database for development (only in debug mode)"""
    if not app.debug:
        return "This endpoint is only available in debug mode", 403

    try:
        # Drop all tables and recreate
        db.drop_all()
        db.create_all()

        # Reinitialize with sample data
        with app.app_context():
            # Create teams
            dev_team = Team(name="Development Team")
            hr_team = Team(name="HR Team")
            qa_team = Team(name="QA Team")
            db.session.add_all([dev_team, hr_team, qa_team])
            db.session.commit()

            # Create users
            users = [
                User(username='admin', password=generate_password_hash(
                    'admin123'), role='admin', team_id=dev_team.id),
                User(username='employee', password=generate_password_hash(
                    'employee123'), role='user', team_id=dev_team.id),
                User(username='john', password=generate_password_hash(
                    'john123'), role='user', team_id=dev_team.id),
                User(username='jane', password=generate_password_hash(
                    'jane123'), role='user', team_id=qa_team.id),
                User(username='hr_manager', password=generate_password_hash(
                    'hr123'), role='admin', team_id=hr_team.id),
            ]
            db.session.add_all(users)
            db.session.commit()

            # Add sample leave history for testing
            sample_leaves = [
                Leave(user_id=2, start_date=datetime(2024, 5, 1).date(), end_date=datetime(2024, 5, 3).date(),
                      reason="Sick leave - flu", status="Approved", decision_reason="Standard sick leave approval"),
                Leave(user_id=2, start_date=datetime(2024, 4, 15).date(), end_date=datetime(2024, 4, 20).date(),
                      reason="Vacation with family", status="Approved", decision_reason="AI Decision: Low business impact, good team coverage"),
                Leave(user_id=3, start_date=datetime(2024, 5, 10).date(), end_date=datetime(2024, 5, 25).date(),
                      reason="Medical procedure and recovery", status="Escalate", decision_reason="AI Escalation: Extended medical leave requires HR review"),
                Leave(user_id=4, start_date=datetime(2024, 5, 5).date(), end_date=datetime(2024, 5, 7).date(),
                      reason="Personal emergency", status="Approved", decision_reason="AI Decision: Emergency situation, high confidence approval"),
            ]
            db.session.add_all(sample_leaves)
            db.session.commit()

        return "Database reset successfully with sample data for Agents SDK testing"
    except Exception as e:
        logger.error(f"Database reset failed: {e}")
        return f"Database reset failed: {str(e)}", 500

# Initialize database tables

# Remove the @app.before_first_request decorator and function
# Replace with a standalone initialization function


def initialize_app():
    """Initialize the application and database tables"""
    try:
        with app.app_context():
            db.create_all()
            logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise


# Complete main function
if __name__ == '__main__':
    # Ensure log directory exists
    import os
    log_dir = os.path.dirname(os.path.abspath('app.log'))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Initialize application and database
    try:
        initialize_app()  # Call the new initialization function
    except Exception as e:
        logger.error(f"Application initialization failed: {e}")
        print(f"❌ Application initialization failed: {e}")
        exit(1)

    # Initialize database with sample data if it doesn't exist
    with app.app_context():
        try:
            # Check if we have any data, if not create sample data
            if not User.query.first():
                logger.info(
                    "📊 No existing data found, creating sample data...")

                # Create teams
                dev_team = Team(name="Development Team")
                hr_team = Team(name="HR Team")
                qa_team = Team(name="QA Team")
                db.session.add_all([dev_team, hr_team, qa_team])
                db.session.commit()

                # Create users
                users = [
                    User(username='admin', password=generate_password_hash(
                        'admin123'), role='admin', team_id=dev_team.id),
                    User(username='employee', password=generate_password_hash(
                        'employee123'), role='user', team_id=dev_team.id),
                    User(username='john', password=generate_password_hash(
                        'john123'), role='user', team_id=dev_team.id),
                    User(username='jane', password=generate_password_hash(
                        'jane123'), role='user', team_id=qa_team.id),
                    User(username='hr_manager', password=generate_password_hash(
                        'hr123'), role='admin', team_id=hr_team.id),
                ]
                db.session.add_all(users)
                db.session.commit()

                logger.info("✅ Sample data created successfully")
                print("\n" + "="*60)
                print("🎉 LEAVE MANAGEMENT SYSTEM - AGENTS SDK VERSION")
                print("="*60)
                print("📋 Sample Login Credentials:")
                print("   Admin: admin/admin123")
                print("   Employee: employee/employee123")
                print("   HR Manager: hr_manager/hr123")
                print("="*60)
                print("🤖 AI Features Available:")
                print("   • Intelligent leave decision making")
                print("   • Automatic escalation detection")
                print("   • Business impact analysis")
                print("   • Team availability assessment")
                print("   • Historical pattern analysis")
                print("="*60)
                print("🔧 Test Endpoints:")
                print("   • /test-agents-sdk - Test AI scenarios")
                print("   • /test-escalation - Test escalation logic")
                print("   • /agent-analytics - View AI analytics")
                print("   • /ai-performance - AI performance metrics")
                print("   • /health - System health check")
                print("="*60)
            else:
                logger.info(
                    "📊 Existing data found, skipping sample data creation")

        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            print(f"❌ Database initialization failed: {e}")

    # Set Flask configuration for development
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

    # Run the application
    try:
        print(f"\n🌐 Starting server on http://127.0.0.1:5000")
        print("🛑 Press Ctrl+C to stop the server")

        app.run(
            debug=True,
            host='127.0.0.1',
            port=5000,
            threaded=True,  # Enable threading for better performance with async agents
            use_reloader=True,
            use_debugger=True
        )
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
        print("\n👋 Server stopped. Goodbye!")
    except Exception as e:
        logger.error(f"Server startup error: {e}")
        print(f"❌ Server failed to start: {e}")
    finally:
        # Cleanup
        logger.info("🧹 Cleaning up resources...")
        # Close any open database connections
        db.session.remove()

# @app.before_first_request
# def create_tables():
#     """Create database tables before first request"""
#     try:
#         db.create_all()
#         logger.info("Database tables created successfully")
#     except Exception as e:
#         logger.error(f"Error creating database tables: {e}")


# # Complete main function
# if __name__ == '__main__':
#     # Ensure log directory exists
#     import os
#     log_dir = os.path.dirname(os.path.abspath('app.log'))
#     if not os.path.exists(log_dir):
#         os.makedirs(log_dir)

#     # Initialize database if it doesn't exist
#     with app.app_context():
#         try:
#             db.create_all()
#             logger.info("🚀 Starting Leave Management System with Agents SDK")
#             logger.info("🤖 AI-powered leave decision system initialized")

#             # Check if we have any data, if not create sample data
#             if not User.query.first():
#                 logger.info(
#                     "📊 No existing data found, creating sample data...")

#                 # Create teams
#                 dev_team = Team(name="Development Team")
#                 hr_team = Team(name="HR Team")
#                 qa_team = Team(name="QA Team")
#                 db.session.add_all([dev_team, hr_team, qa_team])
#                 db.session.commit()

#                 # Create users
#                 users = [
#                     User(username='admin', password=generate_password_hash(
#                         'admin123'), role='admin', team_id=dev_team.id),
#                     User(username='employee', password=generate_password_hash(
#                         'employee123'), role='user', team_id=dev_team.id),
#                     User(username='john', password=generate_password_hash(
#                         'john123'), role='user', team_id=dev_team.id),
#                     User(username='jane', password=generate_password_hash(
#                         'jane123'), role='user', team_id=qa_team.id),
#                     User(username='hr_manager', password=generate_password_hash(
#                         'hr123'), role='admin', team_id=hr_team.id),
#                 ]
#                 db.session.add_all(users)
#                 db.session.commit()

#                 logger.info("✅ Sample data created successfully")
#                 print("\n" + "="*60)
#                 print("🎉 LEAVE MANAGEMENT SYSTEM - AGENTS SDK VERSION")
#                 print("="*60)
#                 print("📋 Sample Login Credentials:")
#                 print("   Admin: admin/admin123")
#                 print("   Employee: employee/employee123")
#                 print("   HR Manager: hr_manager/hr123")
#                 print("="*60)
#                 print("🤖 AI Features Available:")
#                 print("   • Intelligent leave decision making")
#                 print("   • Automatic escalation detection")
#                 print("   • Business impact analysis")
#                 print("   • Team availability assessment")
#                 print("   • Historical pattern analysis")
#                 print("="*60)
#                 print("🔧 Test Endpoints:")
#                 print("   • /test-agents-sdk - Test AI scenarios")
#                 print("   • /test-escalation - Test escalation logic")
#                 print("   • /agent-analytics - View AI analytics")
#                 print("   • /ai-performance - AI performance metrics")
#                 print("   • /health - System health check")
#                 print("="*60)
#             else:
#                 logger.info(
#                     "📊 Existing data found, skipping sample data creation")

#         except Exception as e:
#             logger.error(f"Database initialization error: {e}")
#             print(f"❌ Database initialization failed: {e}")

#     # Set Flask configuration for development
#     app.config['TEMPLATES_AUTO_RELOAD'] = True
#     app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

#     # Run the application
#     try:
#         print(f"\n🌐 Starting server on http://127.0.0.1:5000")
#         print("🛑 Press Ctrl+C to stop the server")

#         app.run(
#             debug=True,
#             host='127.0.0.1',
#             port=5000,
#             threaded=True,  # Enable threading for better performance with async agents
#             use_reloader=True,
#             use_debugger=True
#         )
#     except KeyboardInterrupt:
#         logger.info("🛑 Server stopped by user")
#         print("\n👋 Server stopped. Goodbye!")
#     except Exception as e:
#         logger.error(f"Server startup error: {e}")
#         print(f"❌ Server failed to start: {e}")
#     finally:
#         # Cleanup
#         logger.info("🧹 Cleaning up resources...")
#         # Close any open database connections
#         db.session.remove()
