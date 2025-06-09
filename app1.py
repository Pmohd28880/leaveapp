from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable, Union
from enum import Enum
import asyncio
import json
import logging
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import random

# 1. IMPROVED GOAL AND PLANNING SYSTEM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Goal(Enum):
    PROCESS_LEAVE_REQUEST = "process_leave_request"
    OPTIMIZE_TEAM_COVERAGE = "optimize_team_coverage"
    LEARN_FROM_PATTERNS = "learn_from_patterns"
    INVESTIGATE_ANOMALY = "investigate_anomaly"
    HANDLE_ESCALATION = "handle_escalation"


class AgentState(Enum):
    IDLE = "idle"
    PLANNING = "planning"
    ANALYZING = "analyzing"
    GATHERING_INFO = "gathering_info"
    REASONING = "reasoning"
    DECIDING = "deciding"
    EXECUTING = "executing"
    LEARNING = "learning"
    ERROR = "error"
    REPLANNING = "replanning"


@dataclass
class Plan:
    """Represents an agent's plan to achieve a goal"""
    goal: Goal
    steps: List[Dict[str, Any]] = field(default_factory=list)
    expected_duration: float = 0.0
    confidence: float = 0.0
    fallback_plans: List['Plan'] = field(default_factory=list)

    def add_step(self, action: str, tool: str, params: Dict, expected_outcome: str):
        self.steps.append({
            'action': action,
            'tool': tool,
            'params': params,
            'expected_outcome': expected_outcome,
            'status': 'pending'
        })


@dataclass
class ExecutionResult:
    """Results from executing a plan step"""
    success: bool
    data: Any = None
    confidence: float = 0.0
    errors: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

# 2. ENHANCED MEMORY SYSTEM


@dataclass
class AgentMemory:
    episodic_memory: List[Dict] = field(default_factory=list)
    semantic_memory: Dict = field(default_factory=dict)
    working_memory: Dict = field(default_factory=dict)
    long_term_patterns: Dict = field(default_factory=dict)

    def store_episode(self, episode: Dict):
        """Store experience in episodic memory with automatic indexing"""
        episode['id'] = f"ep_{len(self.episodic_memory)}"
        episode['timestamp'] = datetime.now().isoformat()
        self.episodic_memory.append(episode)
        self._extract_patterns(episode)

    def _extract_patterns(self, episode: Dict):
        """Extract patterns from episodes for semantic memory"""
        decision_type = episode.get('decision_type')
        outcome = episode.get('outcome')
        key = f"{decision_type}_{outcome}" if decision_type and outcome else "unknown"
        if key not in self.semantic_memory:
            self.semantic_memory[key] = {'count': 0, 'confidence': 0.0}
        self.semantic_memory[key]['count'] += 1

# 3. IMPROVED TOOL SYSTEM WITH ASYNC SUPPORT


class AsyncTool(ABC):
    """Enhanced tool interface with async support"""
    @abstractmethod
    async def execute_async(self, **kwargs) -> ExecutionResult:
        pass

    def execute(self, **kwargs) -> ExecutionResult:
        """Sync wrapper for async execution"""
        return asyncio.run(self.execute_async(**kwargs))

    @abstractmethod
    def get_schema(self) -> Dict:
        """Return tool schema for agent planning"""
        pass

    @abstractmethod
    def estimate_execution_time(self, **kwargs) -> float:
        """Estimate execution time for planning"""
        pass


class EnhancedDatabaseTool(AsyncTool):
    """Improved database tool with better error handling and caching"""

    def __init__(self):
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes

    async def execute_async(self, **kwargs) -> ExecutionResult:
        try:
            query_type = kwargs.get('query_type')
            cache_key = f"{query_type}_{hash(str(kwargs))}"
            if self._is_cache_valid(cache_key):
                return ExecutionResult(
                    success=True,
                    data=self.cache[cache_key]['data'],
                    confidence=0.9,
                    metadata={'from_cache': True}
                )
            result = await self._execute_query(query_type, **kwargs)
            self.cache[cache_key] = {
                'data': result,
                'timestamp': datetime.now()
            }
            return ExecutionResult(
                success=True,
                data=result,
                confidence=0.95,
                metadata={'cached': True}
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                errors=[str(e)],
                confidence=0.0
            )

    async def _execute_query(self, query_type: str, **kwargs):
        """Execute the actual database query"""
        pass  # Implement actual database logic here

    def _is_cache_valid(self, cache_key: str) -> bool:
        if cache_key not in self.cache:
            return False
        age = (datetime.now() - self.cache[cache_key]['timestamp']).seconds
        return age < self.cache_ttl

    def get_schema(self) -> Dict:
        return {
            "name": "database_query",
            "description": "Query database for leave and user information",
            "parameters": {
                "query_type": {"type": "string", "required": True},
                "user_id": {"type": "integer"},
                "team_id": {"type": "integer"}
            }
        }

    def estimate_execution_time(self, **kwargs) -> float:
        query_type = kwargs.get('query_type', '')
        if 'complex' in query_type:
            return 2.0
        return 0.5

# 4. ENHANCED REASONER


class EnhancedReasoner:
    """Enhanced reasoning engine with multiple reasoning strategies"""

    def __init__(self):
        self.reasoning_strategies = {
            'analytical': self._analytical_reasoning,
            'pattern_based': self._pattern_based_reasoning,
            'rule_based': self._rule_based_reasoning,
            'case_based': self._case_based_reasoning
        }
        self.confidence_threshold = 0.8

    async def reason(self, goal: Goal, data: Dict, memory: AgentMemory, context: Dict = None) -> Dict:
        reasoning_results = {}
        for strategy_name, strategy_func in self.reasoning_strategies.items():
            try:
                result = await strategy_func(goal, data, memory, context)
                reasoning_results[strategy_name] = result
            except Exception as e:
                reasoning_results[strategy_name] = {
                    'success': False,
                    'error': str(e),
                    'confidence': 0.0
                }
        return await self._combine_reasoning_results(reasoning_results)

    async def _analytical_reasoning(self, goal: Goal, data: Dict, memory: AgentMemory, context: Dict) -> Dict:
        if goal == Goal.PROCESS_LEAVE_REQUEST:
            factors = {
                'policy_compliance': data.get('compliance_status', {}).get('compliant', False),
                'team_impact': self._calculate_team_impact(data),
                'user_history': self._analyze_user_history(data.get('user_history', [])),
                'business_context': self._analyze_business_context(data, context)
            }
            weights = {'policy_compliance': 0.4, 'team_impact': 0.3,
                       'user_history': 0.2, 'business_context': 0.1}
            score = sum(factors[key] * weights[key]
                        for key in factors if key in weights)
            decision = 'Approved' if score > 0.6 else 'Denied' if score < 0.4 else 'Escalate'
            return {
                'success': True,
                'decision': decision,
                'confidence': min(abs(score - 0.5) * 2, 1.0),
                'reasoning': f"Analytical score: {score:.2f}",
                'factors': factors
            }
        return {'success': False, 'error': 'Goal not supported for analytical reasoning'}

    async def _pattern_based_reasoning(self, goal: Goal, data: Dict, memory: AgentMemory, context: Dict) -> Dict:
        similar_cases = self._find_similar_cases(data, memory)
        if not similar_cases:
            return {'success': False, 'error': 'No similar patterns found'}
        successful_outcomes = [
            case for case in similar_cases if case.get('success', False)]
        success_rate = len(successful_outcomes) / \
            len(similar_cases) if similar_cases else 0
        success_patterns = self._extract_success_patterns(successful_outcomes)
        decision = 'Approved' if success_rate > 0.7 else 'Denied' if success_rate < 0.3 else 'Escalate'
        return {
            'success': True,
            'decision': decision,
            'confidence': abs(success_rate - 0.5) * 2,
            'reasoning': f"Pattern analysis: {len(similar_cases)} similar cases, {success_rate:.1%} success rate",
            'patterns': success_patterns
        }

    async def _rule_based_reasoning(self, goal: Goal, data: Dict, memory: AgentMemory, context: Dict) -> Dict:
        rules = [
            {'condition': lambda d: d.get('compliance_status', {}).get(
                'violations', []), 'action': 'Denied', 'priority': 1, 'reason': 'Policy violation'},
            {'condition': lambda d: d.get(
                'risk_level') == 'high', 'action': 'Escalate', 'priority': 2, 'reason': 'High risk'},
            {'condition': lambda d: d.get('team_availability', {}).get(
                'available_members', 10) < 2, 'action': 'Denied', 'priority': 3, 'reason': 'Insufficient team coverage'},
            {'condition': lambda d: d.get('user_history', {}).get(
                'approval_rate', 1.0) > 0.8, 'action': 'Approved', 'priority': 4, 'reason': 'Good user history'}
        ]
        for rule in sorted(rules, key=lambda r: r['priority']):
            try:
                if rule['condition'](data):
                    return {
                        'success': True,
                        'decision': rule['action'],
                        'confidence': 0.9,
                        'reasoning': f"Rule applied: {rule['reason']}",
                        'rule_priority': rule['priority']
                    }
            except Exception:
                continue
        return {
            'success': True,
            'decision': 'Approved',
            'confidence': 0.5,
            'reasoning': 'No rules triggered, using default approval'
        }

    async def _case_based_reasoning(self, goal: Goal, data: Dict, memory: AgentMemory, context: Dict) -> Dict:
        most_similar_case = self._find_most_similar_case(data, memory)
        if not most_similar_case:
            return {'success': False, 'error': 'No similar cases found'}
        similarity_score = most_similar_case.get('similarity', 0.0)
        if similarity_score < 0.6:
            return {'success': False, 'error': 'No sufficiently similar cases'}
        adapted_decision = self._adapt_case_solution(most_similar_case, data)
        return {
            'success': True,
            'decision': adapted_decision['decision'],
            'confidence': similarity_score * adapted_decision['confidence'],
            'reasoning': f"Case-based: {similarity_score:.1%} similar to case {most_similar_case.get('id')}",
            'similar_case': most_similar_case
        }

    async def _combine_reasoning_results(self, results: Dict) -> Dict:
        successful_results = {k: v for k,
                              v in results.items() if v.get('success', False)}
        if not successful_results:
            return {
                'decision': 'Escalate',
                'confidence': 0.0,
                'reasoning': 'All reasoning strategies failed',
                'strategy_results': results
            }
        strategy_weights = {
            'analytical': 0.4, 'pattern_based': 0.3, 'rule_based': 0.2, 'case_based': 0.1}
        decision_scores = {'Approved': 0, 'Denied': 0, 'Escalate': 0}
        total_confidence = 0
        for strategy, result in successful_results.items():
            weight = strategy_weights.get(strategy, 0.1)
            confidence = result.get('confidence', 0.0)
            decision = result.get('decision', 'Escalate')
            decision_scores[decision] += weight * confidence
            total_confidence += weight * confidence
        final_decision = max(decision_scores, key=decision_scores.get)
        final_confidence = total_confidence / \
            len(successful_results) if successful_results else 0.5
        return {
            'decision': final_decision,
            'confidence': final_confidence,
            'reasoning': f"Combined reasoning from {len(successful_results)} strategies",
            'strategy_results': results,
            'decision_scores': decision_scores
        }

    def _calculate_team_impact(self, data: Dict) -> float:
        team_data = data.get('team_availability', {})
        available = team_data.get('available_members', 5)
        total = team_data.get('total_members', 5)
        if total == 0:
            return 0.0
        availability_ratio = available / total
        return max(0, 1 - availability_ratio)

    def _analyze_user_history(self, history: List[Dict]) -> float:
        if not history:
            return 0.5
        approved = sum(1 for h in history if h.get('status') == 'Approved')
        total = len(history)
        return approved / total if total > 0 else 0.5

    def _analyze_business_context(self, data: Dict, context: Dict) -> float:
        if not context:
            return 0.5
        workload = context.get('team_workload', 'normal')
        deadlines = context.get('upcoming_deadlines', [])
        score = 0.5
        if workload == 'high':
            score -= 0.2
        elif workload == 'low':
            score += 0.2
        if len(deadlines) > 2:
            score -= 0.1
        return max(0, min(1, score))

    def _find_similar_cases(self, data: Dict, memory: AgentMemory) -> List[Dict]:
        similar_cases = []
        for episode in memory.episodic_memory:
            similarity = self._calculate_similarity(data, episode)
            if similarity > 0.5:
                episode['similarity'] = similarity
                similar_cases.append(episode)
        return sorted(similar_cases, key=lambda x: x['similarity'], reverse=True)

    def _find_most_similar_case(self, data: Dict, memory: AgentMemory) -> Optional[Dict]:
        similar_cases = self._find_similar_cases(data, memory)
        return similar_cases[0] if similar_cases else None

    def _calculate_similarity(self, data1: Dict, data2: Dict) -> float:
        common_keys = set(data1.keys()) & set(data2.keys())
        if not common_keys:
            return 0.0
        similarity_sum = sum(
            1 for key in common_keys if data1[key] == data2[key])
        return similarity_sum / len(common_keys)

    def _extract_success_patterns(self, successful_cases: List[Dict]) -> Dict:
        patterns = {}
        for case in successful_cases:
            for key, value in case.items():
                if key not in patterns:
                    patterns[key] = {}
                patterns[key][value] = patterns[key].get(value, 0) + 1
        return patterns

    def _adapt_case_solution(self, similar_case: Dict, current_data: Dict) -> Dict:
        base_decision = similar_case.get('decision', 'Escalate')
        base_confidence = similar_case.get('confidence', 0.5)
        similarity = similar_case.get('similarity', 0.5)
        return {
            'decision': base_decision,
            'confidence': base_confidence * similarity
        }

# 5. ENHANCED AGENT WITH BETTER PLANNING


class EnhancedAutonomousAgent:
    def __init__(self):
        self.state = AgentState.IDLE
        self.current_goal: Optional[Goal] = None
        self.current_plan: Optional[Plan] = None
        self.memory = AgentMemory()
        self.tools: Dict[str, AsyncTool] = {}
        self.planner = AgentPlanner()
        self.reasoner = EnhancedReasoner()
        self.executor = PlanExecutor()
        self.performance_metrics = {
            'decisions_made': 0,
            'success_rate': 0.0,
            'avg_confidence': 0.0,
            'avg_processing_time': 0.0
        }

    async def _handle_error(self, error: Exception, request: Any, context: Dict = None) -> Dict:
        self.state = AgentState.ERROR
        error_result = {
            'status': 'Error',
            'error': str(error),
            'confidence': 0.0,
            'success': False,
            'agent_reasoning': f'Processing failed: {str(error)}'
        }
        logging.error(f"Agent error: {error}")
        error_episode = {
            'request_type': type(request).__name__,
            'goal': self.current_goal.value if self.current_goal else 'unknown',
            'error': str(error),
            'success': False,
            'timestamp': datetime.now().isoformat()
        }
        self.memory.store_episode(error_episode)
        return error_result

    async def process_request(self, request: Any, context: Dict = None) -> Dict:
        try:
            self.current_goal = self._determine_goal(request)
            self.state = AgentState.PLANNING
            self.current_plan = await self.planner.create_plan(
                goal=self.current_goal,
                request=request,
                context=context,
                available_tools=self.tools,
                memory=self.memory
            )
            result = await self.executor.execute_plan(
                plan=self.current_plan,
                tools=self.tools,
                memory=self.memory
            )
            await self._learn_from_outcome(request, result)
            self._update_metrics(result)
            return result
        except Exception as e:
            return await self._handle_error(e, request, context)

    def _determine_goal(self, request: Any) -> Goal:
        if hasattr(request, 'urgency') and request.urgency == 'high':
            return Goal.HANDLE_ESCALATION
        elif hasattr(request, 'duration_days') and request.duration_days > 14:
            return Goal.OPTIMIZE_TEAM_COVERAGE
        else:
            return Goal.PROCESS_LEAVE_REQUEST

    def _update_metrics(self, result: Dict):
        self.performance_metrics['decisions_made'] += 1
        if result.get('success', False):
            alpha = 0.1
            current_success = 1.0
            self.performance_metrics['success_rate'] = (
                alpha * current_success +
                (1 - alpha) * self.performance_metrics['success_rate']
            )
        if 'confidence' in result:
            alpha = 0.1
            self.performance_metrics['avg_confidence'] = (
                alpha * result['confidence'] +
                (1 - alpha) * self.performance_metrics['avg_confidence']
            )

    async def _learn_from_outcome(self, request: Any, result: Dict):
        episode = {
            'request_type': type(request).__name__,
            'goal': self.current_goal.value,
            'plan_steps': len(self.current_plan.steps) if self.current_plan else 0,
            'outcome': result.get('status'),
            'confidence': result.get('confidence', 0.0),
            'success': result.get('success', False),
            'processing_time': result.get('processing_time', 0.0)
        }
        self.memory.store_episode(episode)
        await self._update_patterns(episode)

    async def _update_patterns(self, episode: Dict):
        pattern_key = f"{episode['request_type']}_{episode['goal']}"
        if pattern_key not in self.memory.long_term_patterns:
            self.memory.long_term_patterns[pattern_key] = {
                'success_rate': 0.0,
                'avg_confidence': 0.0,
                'common_failures': [],
                'optimal_strategies': []
            }
        pattern = self.memory.long_term_patterns[pattern_key]
        alpha = 0.1
        pattern['success_rate'] = (
            alpha * float(episode['success']) +
            (1 - alpha) * pattern['success_rate']
        )
        pattern['avg_confidence'] = (
            alpha * episode['confidence'] +
            (1 - alpha) * pattern['avg_confidence']
        )

# 6. ENHANCED PLANNER


class AgentPlanner:
    async def create_plan(self, goal: Goal, request: Any, context: Dict, available_tools: Dict, memory: AgentMemory) -> Plan:
        plan = Plan(goal=goal)
        if goal == Goal.PROCESS_LEAVE_REQUEST:
            plan = await self._plan_leave_processing(request, available_tools, memory)
        elif goal == Goal.OPTIMIZE_TEAM_COVERAGE:
            plan = await self._plan_team_optimization(request, available_tools, memory)
        plan.fallback_plans = await self._create_fallback_plans(plan, available_tools)
        return plan

    async def _plan_leave_processing(self, request: Any, tools: Dict, memory: AgentMemory) -> Plan:
        plan = Plan(goal=Goal.PROCESS_LEAVE_REQUEST)
        plan.add_step(
            action="gather_user_history",
            tool="database",
            params={"query_type": "user_leave_history",
                    "user_id": request.user_id},
            expected_outcome="user_leave_patterns"
        )
        plan.add_step(
            action="check_compliance",
            tool="database",
            params={"query_type": "policy_check", "leave_request": request},
            expected_outcome="compliance_status"
        )
        plan.add_step(
            action="assess_risk",
            tool="database",
            params={"query_type": "team_availability"},
            expected_outcome="risk_level"
        )
        plan.expected_duration = sum(
            tools[step['tool']].estimate_execution_time(**step['params'])
            for step in plan.steps if step['tool'] in tools
        )
        return plan

    async def _plan_team_optimization(self, request: Any, tools: Dict, memory: AgentMemory) -> Plan:
        plan = Plan(goal=Goal.OPTIMIZE_TEAM_COVERAGE)
        plan.add_step(
            action="analyze_team_capacity",
            tool="database",
            params={"query_type": "team_availability"},
            expected_outcome="team_capacity"
        )
        return plan

    async def _create_fallback_plans(self, plan: Plan, available_tools: Dict) -> List[Plan]:
        fallback_plans = []
        escalation_plan = Plan(goal=plan.goal)
        escalation_plan.add_step(
            action="escalate_decision",
            tool="database",
            params={"query_type": "escalation",
                    "reason": "primary_plan_failed"},
            expected_outcome="escalated_to_human"
        )
        fallback_plans.append(escalation_plan)
        return fallback_plans

# 7. ENHANCED EXECUTOR WITH PARALLEL PROCESSING


class PlanExecutor:
    async def execute_plan(self, plan: Plan, tools: Dict, memory: AgentMemory) -> Dict:
        start_time = datetime.now()
        results = {}
        try:
            parallel_groups = self._group_parallel_steps(plan.steps)
            for group in parallel_groups:
                group_results = await self._execute_parallel_group(group, tools)
                results.update(group_results)
                if self._should_replan(group_results, plan):
                    return await self._replan_and_execute(plan, tools, memory, results)
            final_result = await self._synthesize_results(results, plan.goal)
            final_result['processing_time'] = (
                datetime.now() - start_time).total_seconds()
            return final_result
        except Exception as e:
            for fallback_plan in plan.fallback_plans:
                try:
                    return await self.execute_plan(fallback_plan, tools, memory)
                except:
                    continue
            return {
                'status': 'Error',
                'error': str(e),
                'confidence': 0.0,
                'processing_time': (datetime.now() - start_time).total_seconds()
            }

    def _group_parallel_steps(self, steps: List[Dict]) -> List[List[Dict]]:
        groups = []
        current_group = []
        for step in steps:
            if len(current_group) < 3:
                current_group.append(step)
            else:
                groups.append(current_group)
                current_group = [step]
        if current_group:
            groups.append(current_group)
        return groups

    async def _execute_parallel_group(self, group: List[Dict], tools: Dict) -> Dict:
        tasks = []
        for step in group:
            tool_name = step['tool']
            if tool_name in tools:
                task = tools[tool_name].execute_async(**step['params'])
                tasks.append((step['action'], task))
        results = {}
        for action, task in tasks:
            try:
                result = await task
                results[action] = result
            except Exception as e:
                results[action] = ExecutionResult(
                    success=False, errors=[str(e)])
        return results

    async def _synthesize_results(self, results: Dict, goal: Goal) -> Dict:
        """Synthesize final result from all step results"""
        successful_ops = sum(1 for r in results.values()
                             if getattr(r, 'success', False))
        total_ops = len(results)

    # ADD THIS: Don't synthesize if we should call reasoner instead
        if successful_ops >= total_ops * 0.5:  # If most operations succeeded
            # Extract data for reasoner
            reasoner_data = {}
            for action, result in results.items():
                if hasattr(result, 'data'):
                    reasoner_data[action] = result.data

        # Call the actual reasoner (you need to pass it from the executor)
        # This requires modifying the executor constructor to accept a reasoner
            reasoner = TransparentReasoner()  # or get from constructor
            reasoning_result = await reasoner.reason(goal, reasoner_data, None)

            return {
                'status': reasoning_result.get('decision', 'Escalate'),
                'confidence': reasoning_result.get('confidence', 0.5),
                'success': reasoning_result.get('decision') in ['Approved'],
                'agent_reasoning': reasoning_result.get('detailed_explanation', 'Reasoning completed'),
                'results': results,
                'detailed_explanation': reasoning_result.get('detailed_explanation')
            }

        return {
            'status': status,
            'confidence': final_confidence,
            'success': status in ['Approved'],
            'agent_reasoning': detailed_explanation,
            'results': results,
            'detailed_explanation': detailed_explanation
        }

    def _should_replan(self, group_results: Dict, plan: Plan) -> bool:
        failed_ops = sum(1 for r in group_results.values()
                         if not getattr(r, 'success', True))
    # CHANGE THIS LINE - it was too aggressive
        # Only replan if 80%+ fail, not 50%
        return failed_ops > len(group_results) * 0.8

    async def _replan_and_execute(self, plan: Plan, tools: Dict, memory, partial_results: Dict) -> Dict:
        return {
            'status': 'Escalate',
            'confidence': 0.3,
            'success': False,
            'agent_reasoning': 'Replanning required due to execution issues',
            'processing_time': 0.0
        }

# 8. DETAILED DATABASE TOOL


class DetailedDatabaseTool(AsyncTool):
    def __init__(self, team_members, user_id=None):
        self.team_members = team_members
        self.user_id = user_id

    async def execute_async(self, **kwargs) -> ExecutionResult:
        query_type = kwargs.get('query_type', 'unknown')
        print(f"🔍 Executing {query_type} query...")
        if query_type == 'user_leave_history':
            user_id = kwargs.get('user_id', self.user_id)
            history_data = self._analyze_user_history(user_id)
            print(f"📊 User {user_id} history: {history_data}")
            return ExecutionResult(
                success=True,
                data=history_data,
                confidence=0.95,
                metadata={'query_type': query_type, 'user_id': user_id}
            )
        elif query_type == 'team_availability':
            team_data = self._analyze_team_capacity()
            print(f"👥 Team analysis: {team_data}")
            return ExecutionResult(
                success=True,
                data=team_data,
                confidence=0.9,
                metadata={'query_type': query_type}
            )
        elif query_type == 'policy_check':
            leave_request = kwargs.get('leave_request')
            policy_data = self._check_policies(leave_request)
            print(f"📋 Policy check: {policy_data}")
            return ExecutionResult(
                success=True,
                data=policy_data,
                confidence=0.95,
                metadata={'query_type': query_type}
            )
        return ExecutionResult(success=False, errors=[f"Unknown query type: {query_type}"])

    def _analyze_user_history(self, user_id):
        user_profiles = {
            'good_employee': {'approval_rate': 0.9, 'total_leaves': 8, 'avg_duration': 3, 'reliability_score': 0.85},
            'average_employee': {'approval_rate': 0.7, 'total_leaves': 12, 'avg_duration': 5, 'reliability_score': 0.65},
            'problematic_employee': {'approval_rate': 0.4, 'total_leaves': 20, 'avg_duration': 8, 'reliability_score': 0.3}
        }
        if user_id % 3 == 0:
            profile = user_profiles['good_employee']
            profile_type = 'good_employee'
        elif user_id % 3 == 1:
            profile = user_profiles['average_employee']
            profile_type = 'average_employee'
        else:
            profile = user_profiles['problematic_employee']
            profile_type = 'problematic_employee'
        return {
            **profile,
            'profile_type': profile_type,
            'last_leave_date': '2024-05-15',
            'consecutive_approvals': profile['approval_rate'] * 5
        }

    def _analyze_team_capacity(self):
        total_members = len(self.team_members) if self.team_members else 5
        random.seed(42)
        currently_on_leave = random.randint(0, max(1, total_members // 3))
        available_members = total_members - currently_on_leave
        workload_level = random.choice(['low', 'normal', 'high', 'critical'])
        upcoming_deadlines = random.randint(0, 3)
        capacity_score = self._calculate_capacity_score(
            available_members, total_members, workload_level, upcoming_deadlines)
        return {
            'total_members': total_members,
            'available_members': available_members,
            'currently_on_leave': currently_on_leave,
            'workload_level': workload_level,
            'upcoming_deadlines': upcoming_deadlines,
            'capacity_score': capacity_score,
            'capacity_status': self._get_capacity_status(capacity_score)
        }

    def _calculate_capacity_score(self, available, total, workload, deadlines):
        base_score = available / total if total > 0 else 0
        workload_multipliers = {'low': 1.2,
                                'normal': 1.0, 'high': 0.7, 'critical': 0.4}
        workload_adjustment = workload_multipliers.get(workload, 1.0)
        deadline_penalty = deadlines * 0.1
        final_score = max(
            0, min(1, base_score * workload_adjustment - deadline_penalty))
        return round(final_score, 2)

    def _get_capacity_status(self, score):
        if score >= 0.8:
            return 'excellent'
        elif score >= 0.6:
            return 'good'
        elif score >= 0.4:
            return 'moderate'
        elif score >= 0.2:
            return 'limited'
        else:
            return 'critical'

    def _check_policies(self, leave_request):
        violations = []
        compliance_score = 1.0
        if hasattr(leave_request, 'start_date'):
            try:
                start_date = datetime.strptime(
                    leave_request.start_date, '%Y-%m-%d')
                days_notice = (start_date - datetime.now()).days
                if days_notice < 3:
                    violations.append(
                        f"Insufficient notice: {days_notice} days (minimum 3 required)")
                    compliance_score -= 0.3
                elif days_notice < 7:
                    violations.append(
                        f"Short notice: {days_notice} days (recommended 7+ days)")
                    compliance_score -= 0.1
            except:
                violations.append("Invalid date format")
                compliance_score -= 0.2
        if hasattr(leave_request, 'duration_days'):
            if leave_request.duration_days > 14:
                violations.append(
                    f"Extended leave: {leave_request.duration_days} days (>14 days requires special approval)")
                compliance_score -= 0.2
            elif leave_request.duration_days > 7:
                violations.append(
                    f"Long leave: {leave_request.duration_days} days (may impact team)")
                compliance_score -= 0.1
        if hasattr(leave_request, 'reason'):
            suspicious_reasons = ['personal', 'other', 'none', '']
            if leave_request.reason.lower().strip() in suspicious_reasons:
                violations.append("Vague reason provided")
                compliance_score -= 0.1
        compliance_score = max(0, compliance_score)
        return {
            'compliant': len(violations) == 0,
            'violations': violations,
            'compliance_score': round(compliance_score, 2),
            'policy_status': 'compliant' if compliance_score >= 0.8 else 'minor_issues' if compliance_score >= 0.6 else 'major_issues'
        }

    def get_schema(self) -> Dict:
        return {"name": "detailed_database", "description": "Detailed database with business logic"}

    def estimate_execution_time(self, **kwargs) -> float:
        return 0.2

# 9. TRANSPARENT REASONER


class TransparentReasoner(EnhancedReasoner):
    async def reason(self, goal: Goal, data: Dict, memory: AgentMemory, context: Dict = None) -> Dict:
        print(f"\n🧠 REASONING PROCESS FOR {goal.value}")
        print("=" * 50)
        result = await super().reason(goal, data, memory, context)
        explanation = await self._generate_detailed_explanation(goal, data, result)
        result['detailed_explanation'] = explanation
        print(f"🎯 FINAL DECISION: {result.get('decision', 'Unknown')}")
        print(f"🎲 CONFIDENCE: {result.get('confidence', 0):.0%}")
        print(f"💡 EXPLANATION: {explanation}")
        print("=" * 50)
        return result

    async def _generate_detailed_explanation(self, goal: Goal, data: Dict, result: Dict) -> str:
        decision = result.get('decision', 'Unknown')
        explanation_parts = []
        if 'user_leave_history' in str(data):
            user_data = self._extract_user_data(data)
            if user_data:
                explanation_parts.append(
                    f"User profile: {user_data.get('profile_type', 'unknown')} with {user_data.get('approval_rate', 0):.0%} approval rate")
        if 'team_availability' in str(data):
            team_data = self._extract_team_data(data)
            if team_data:
                explanation_parts.append(
                    f"Team capacity: {team_data.get('capacity_status', 'unknown')} ({team_data.get('available_members', 0)}/{team_data.get('total_members', 0)} available)")
        if 'policy_check' in str(data):
            policy_data = self._extract_policy_data(data)
            if policy_data:
                status = policy_data.get('policy_status', 'unknown')
                explanation_parts.append(f"Policy compliance: {status}")
                if policy_data.get('violations'):
                    explanation_parts.append(
                        f"Issues: {'; '.join(policy_data['violations'])}")
        if decision == 'Approved':
            explanation_parts.append("✅ All criteria met for approval")
        elif decision == 'Denied':
            explanation_parts.append("❌ Critical issues prevent approval")
        else:
            explanation_parts.append("⚠️ Requires human review")
        return " | ".join(explanation_parts) if explanation_parts else "Decision based on available data"

    def _extract_user_data(self, data):
        for key, value in data.items():
            if hasattr(value, 'data') and isinstance(value.data, dict) and 'profile_type' in value.data:
                return value.data
        return None

    def _extract_team_data(self, data):
        for key, value in data.items():
            if hasattr(value, 'data') and isinstance(value.data, dict) and 'capacity_status' in value.data:
                return value.data
        return None

    def _extract_policy_data(self, data):
        for key, value in data.items():
            if hasattr(value, 'data') and isinstance(value.data, dict) and 'policy_status' in value.data:
                return value.data
        return None

# 10. COMPREHENSIVE PLANNER


class ComprehensivePlanner(AgentPlanner):
    async def _plan_leave_processing(self, request: Any, tools: Dict, memory: AgentMemory) -> Plan:
        print(f"📋 CREATING COMPREHENSIVE ANALYSIS PLAN")
        print(f"   User: {getattr(request, 'user_id', 'unknown')}")
        print(
            f"   Duration: {getattr(request, 'duration_days', 'unknown')} days")
        print(f"   Reason: {getattr(request, 'reason', 'unknown')}")
        plan = Plan(goal=Goal.PROCESS_LEAVE_REQUEST)
        plan.add_step(
            action="analyze_user_history",
            tool="database",
            params={"query_type": "user_leave_history",
                    "user_id": getattr(request, 'user_id', 1)},
            expected_outcome="user_reliability_profile"
        )
        plan.add_step(
            action="check_team_capacity",
            tool="database",
            params={"query_type": "team_availability"},
            expected_outcome="team_capacity_analysis"
        )
        plan.add_step(
            action="validate_policies",
            tool="database",
            params={"query_type": "policy_check", "leave_request": request},
            expected_outcome="policy_compliance_report"
        )
        return plan

# 11. INTEGRATION FUNCTIONS


async def process_leave_with_agentic_ai(form_data, team_members):
    """Integration function with detailed logging"""
    print(f"\n🤖 STARTING AGENTIC AI ANALYSIS")
    print(
        f"📅 Leave period: {form_data['start_date']} to {form_data['end_date']}")
    print(f"📝 Reason: {form_data['reason']}")
    print(f"👥 Team size: {len(team_members) if team_members else 'unknown'}")
    try:
        start_date = datetime.strptime(form_data['start_date'], '%Y-%m-%d')
        end_date = datetime.strptime(form_data['end_date'], '%Y-%m-%d')
        duration_days = (end_date - start_date).days + 1

        @dataclass
        class LeaveRequest:
            user_id: int
            reason: str
            start_date: str
            end_date: str
            urgency: str = "normal"
            duration_days: int = 1

        leave_request = LeaveRequest(
            user_id=getattr(current_user, 'id', 1),
            reason=form_data['reason'],
            start_date=form_data['start_date'],
            end_date=form_data['end_date'],
            urgency=form_data.get('urgency', 'normal'),
            duration_days=duration_days
        )

        result = await process_with_transparent_agent(leave_request, team_members)
        return result
    except Exception as e:
        logger.error(f"Error in agentic AI processing: {str(e)}")
        return {
            'status': 'Escalate',
            'confidence': 0.0,
            'success': False,
            'agent_reasoning': f'System error: {str(e)}',
            'processing_time': 0.0,
            'reason': 'Technical issue - requires manual review'
        }


async def process_with_transparent_agent(leave_request, team_members):
    """Process with full transparency"""
    agent = EnhancedAutonomousAgent()
    agent.reasoner = TransparentReasoner()
    agent.planner = ComprehensivePlanner()
    agent.tools['database'] = DetailedDatabaseTool(
        team_members, leave_request.user_id)
    result = await agent.process_request(leave_request)
    result['processing_time'] = result.get(
        'processing_time', 0.5)  # Ensure realistic processing time
    result['confidence'] = result.get('confidence', 0.8)
    result['success'] = result.get('success', True)
    if 'detailed_explanation' in result:
        result['agent_reasoning'] = result['detailed_explanation']
    else:
        result['agent_reasoning'] = 'Standard AI analysis completed'
    result['reason'] = f"AI Decision: {result['agent_reasoning']}"
    print(f"\n✅ ANALYSIS COMPLETE - Decision: {result['status']}")
    return result


async def process_agentic_request(leave_request, team_members):
    """Process leave request using agentic AI"""
    try:
        agent = EnhancedAutonomousAgent()
        agent.tools['database'] = MockDatabaseTool(team_members)
        result = await agent.process_request(leave_request)
        result.setdefault('processing_time', 0.5)
        result.setdefault('confidence', 0.8)
        result.setdefault('success', True)
        result.setdefault('agent_reasoning',
                          'Processed using agentic AI system')
        result.setdefault(
            'reason', 'AI decision based on policy and team availability')
        return result
    except Exception as e:
        logger.error(f"Error in agentic processing: {str(e)}")
        return {
            'status': 'Escalate',
            'confidence': 0.0,
            'success': False,
            'agent_reasoning': f'Processing failed: {str(e)}',
            'processing_time': 0.0,
            'reason': 'System error - requires manual review'
        }


class MockDatabaseTool(AsyncTool):
    def __init__(self, team_members):
        self.team_members = team_members

    async def execute_async(self, **kwargs) -> ExecutionResult:
        query_type = kwargs.get('query_type', 'unknown')
        if query_type == 'user_leave_history':
            return ExecutionResult(
                success=True,
                data={'approval_rate': 0.8, 'total_leaves': 5},
                confidence=0.9
            )
        elif query_type == 'team_availability':
            available_members = max(1, len(self.team_members) - 1)
            return ExecutionResult(
                success=True,
                data={'available_members': available_members,
                      'total_members': len(self.team_members)},
                confidence=0.95
            )
        else:
            return ExecutionResult(
                success=True,
                data={'status': 'ok'},
                confidence=0.7
            )

    def get_schema(self) -> Dict:
        return {"name": "mock_database", "description": "Mock database for demo"}

    def estimate_execution_time(self, **kwargs) -> float:
        return 0.1

# 12. USAGE EXAMPLE


class MockUser:
    id = 123


current_user = MockUser()


async def main():
    """Example usage of enhanced agent with detailed logging"""
    try:
        # Positive scenario
        form_data = {
            'start_date': '2025-07-01',
            'end_date': '2025-07-07',
            'reason': 'Vacation',
            'urgency': 'normal'
        }
        team_members = ['Alice', 'Bob', 'Charlie', 'David', 'Eve']
        result = await process_leave_with_agentic_ai(form_data, team_members)
        print(f"\n=== Final Result ===")
        print(f"Decision: {result.get('status', 'Unknown')}")
        print(f"Confidence: {result.get('confidence', 0.0):.2%}")
        print(f"Processing time: {result.get('processing_time', 0.0):.2f}s")
        print(
            f"Reasoning: {result.get('agent_reasoning', 'No reasoning provided')}")
        print(f"Reason: {result.get('reason', 'No reason provided')}")

        # Negative scenario
        print("\n=== Testing Negative Scenario ===")
        form_data = {
            'start_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
            'end_date': (datetime.now() + timedelta(days=5)).strftime('%Y-%m-%d'),
            'reason': 'personal',
            'urgency': 'high'
        }
        team_members = ['Alice']
        current_user.id = 5  # problematic_employee
        result = await process_leave_with_agentic_ai(form_data, team_members)
        print(f"\n=== Negative Scenario Result ===")
        print(f"Decision: {result.get('status', 'Unknown')}")
        print(f"Confidence: {result.get('confidence', 0.0):.2%}")
        print(f"Processing time: {result.get('processing_time', 0.0):.2f}s")
        print(
            f"Reasoning: {result.get('agent_reasoning', 'No reasoning provided')}")
        print(f"Reason: {result.get('reason', 'No reason provided')}")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
