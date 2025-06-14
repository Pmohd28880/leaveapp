class AgenticLeaveSystemSDK:
    def __init__(self):
        self.triage_agent = triage_agent  # Placeholder
        self.context_agent = context_agent  # Placeholder
        self.executor = ThreadPoolExecutor(max_workers=3)
        from app import app
        with app.app_context():
            if not db.session:
                db.session = db.create_scoped_session()
                logger.info("Initialized new database session")

    def validate_and_analyze_leave(self, user_id: str, leave_type_id: str, duration: float,
                                   start_date: str, end_date: str, reason: str, fiscal_year: int = None) -> Dict:
        """Validate and analyze leave request using integrated functions"""
        try:
            from app import app
            with app.app_context():
                # Step 1: Run comprehensive validation
                validation_result = get_comprehensive_leave_validation(
                    user_id, start_date, end_date)

                # Step 2: Run comprehensive analysis
                analysis_result = get_comprehensive_leave_analysis(
                    user_id, leave_type_id, duration, start_date, end_date, fiscal_year
                )

                # Step 3: Find similar leave decisions
                precedent_result = get_similar_leave_decisions(
                    reason, duration, user_id, analysis_result['user_info'].get(
                        'team_public_id'), start_date
                )

                # Combine results for AI decision
                ai_decision = {
                    'status': 'Approved',
                    'reason': 'Initial AI evaluation',
                    'validation': validation_result,
                    'analysis': analysis_result,
                    'precedents': precedent_result
                }

                # Decision logic
                if not validation_result.get('overall_valid', True):
                    ai_decision['status'] = 'Rejected'
                    ai_decision[
                        'reason'] = f"Invalid leave request: {validation_result.get('rejection_reasons', [])}"
                elif analysis_result['overall_assessment']['recommendation'] in ['rejected', 'conditional']:
                    ai_decision['status'] = 'Escalate'
                    ai_decision['reason'] = f"High risk: {analysis_result['overall_assessment']['blocking_factors']}"
                elif precedent_result['historical_approval_rate'] < 40:
                    ai_decision['status'] = 'Escalate'
                    ai_decision['reason'] = f"Low historical approval rate: {precedent_result['historical_approval_rate']}%"

                return ai_decision

        except Exception as e:
            logger.error(f"Error in validate_and_analyze_leave: {str(e)}")
            return {
                'status': 'Error',
                'reason': f"Analysis failed: {str(e)}",
                'validation': {},
                'analysis': {},
                'precedents': {}
            }

    def process_leave_decision(self, leave_record, ai_decision: Dict) -> Dict:
        """Process the AI decision and update the leave request status and balance"""
        try:
            from app import app
            with app.app_context():
                # Handle different input types
                if isinstance(leave_record, str):
                    leave = Leave.query.get(leave_record)
                    if not leave:
                        logger.error(
                            f"Leave record with ID {leave_record} not found")
                        return {"error": f"Leave record {leave_record} not found in database"}
                    logger.info(f"Found Leave object with ID: {leave.id}")

                elif isinstance(leave_record, Leave):
                    leave = leave_record
                    logger.info(
                        f"Using existing Leave object with ID: {leave.id}")

                else:
                    leave = self._find_leave_record(leave_record)
                    if not leave:
                        logger.error(
                            "Could not find Leave record for LeaveRequest")
                        return {"error": "Leave request not found in database"}

                # Get balance info before processing
                balance_before = self.get_current_balance_info(
                    leave.user_id, leave.leave_type_id, leave.start_date.year
                )
                logger.info(f"🔍 BALANCE BEFORE PROCESSING: {balance_before}")

                old_status = leave.status
                logger.info(
                    f"Processing leave {leave.public_id} - Old status: {old_status}, New status: {ai_decision['status']}")

                # Use duration for leave days
                leave_days = float(leave.duration) if leave.duration else 0.0
                if leave.is_half_day:
                    leave_days = 0.5 if leave_days == 1.0 else leave_days
                logger.info(f"Leave duration: {leave_days} days")

                # Update leave status based on AI decision
                if ai_decision['status'] == 'Approved':
                    leave.status = 'Approved'
                    leave.approved_at = datetime.now(timezone.utc)
                    leave.decision_reason = ai_decision['reason']
                    self._update_leave_balance_directly(
                        leave, old_status, 'Approved', leave_days)
                    logger.info(
                        f"✅ Leave request {leave.public_id} approved and balance updated")

                elif ai_decision['status'] in ['Denied', 'Rejected']:
                    leave.status = 'Rejected'
                    leave.decision_reason = ai_decision['reason']
                    self._update_leave_balance_directly(
                        leave, old_status, 'Rejected', leave_days)
                    logger.info(
                        f"❌ Leave request {leave.public_id} rejected and balance updated")

                elif ai_decision['status'] == 'Escalate':
                    leave.status = 'Pending'
                    leave.decision_reason = f"Escalated: {ai_decision['reason']}"
                    logger.info(
                        f"⬆️ Leave request {leave.public_id} escalated for human review")

                # Save changes
                db.session.commit()

                # Get balance info after processing
                balance_after = self.get_current_balance_info(
                    leave.user_id, leave.leave_type_id, leave.start_date.year
                )
                logger.info(f"🔍 BALANCE AFTER PROCESSING: {balance_after}")

                return {
                    "success": True,
                    "leave_id": leave.public_id,
                    "new_status": leave.status,
                    "balance_updated": ai_decision['status'] in ['Approved', 'Rejected'],
                    "balance_before": balance_before,
                    "balance_after": balance_after
                }

        except Exception as e:
            db.session.rollback()
            logger.exception(f"Error processing leave decision: {str(e)}")
            return {
                "error": f"Failed to process decision: {str(e)}",
                "success": False,
                "leave_id": None,
                "new_status": None,
                "balance_updated": False,
                "balance_before": {},
                "balance_after": {}
            }

    def _find_leave_record(self, leave_request) -> Leave:
        """Helper method to find Leave record from LeaveRequest"""
        leave = None

        # Strategy 1: By public_id
        if hasattr(leave_request, 'public_id') and leave_request.public_id:
            leave = Leave.query.filter_by(
                public_id=leave_request.public_id).first()
            if leave:
                logger.info(
                    f"Found leave by public_id: {leave_request.public_id}")
                return leave

        # Strategy 2: By user_id, dates, and status
        if hasattr(leave_request, 'user_id'):
            leave = Leave.query.filter_by(
                user_id=leave_request.user_id,
                start_date=leave_request.start_date,
                end_date=leave_request.end_date,
                status='Pending'
            ).order_by(desc(Leave.created_at)).first()
            if leave:
                logger.info(
                    f"Found leave by user_id and dates: {leave.public_id}")
                return leave

        # Strategy 3: Most recent pending leave for user
        if hasattr(leave_request, 'user_id'):
            leave = Leave.query.filter_by(
                user_id=leave_request.user_id,
                status='Pending'
            ).order_by(desc(Leave.created_at)).first()
            if leave:
                logger.info(
                    f"Found most recent pending leave: {leave.public_id}")
                return leave

        return None

    def _update_leave_balance_directly(self, leave: Leave, old_status: str, new_status: str, leave_days: float):
        """Directly update the leave balance in the LeaveBalance table"""
        try:
            leave_balance = LeaveBalance.query.filter_by(
                user_id=leave.user_id,
                leave_type_id=leave.leave_type_id,
                fiscal_year=leave.start_date.year
            ).first()

            if not leave_balance:
                logger.error(
                    f"No leave balance found for user {leave.user_id}, leave type {leave.leave_type_id}")
                return

            # Calculate current remaining balance BEFORE update
            remaining_before = float(
                leave_balance.total_available - leave_balance.total_used)

            logger.info(f"BEFORE UPDATE:")
            logger.info(
                f"  - Total Available: {leave_balance.total_available}")
            logger.info(f"  - Total Used: {leave_balance.total_used}")
            logger.info(f"  - Remaining: {remaining_before}")
            logger.info(f"  - Status Transition: {old_status} → {new_status}")
            logger.info(f"  - Leave Days: {leave_days}")

            # Handle status transitions
            if old_status == 'Pending' and new_status == 'Approved':
                leave_balance.total_used += leave_days
                logger.info(
                    f"  - Action: Added {leave_days} days to total_used")

            elif old_status == 'Pending' and new_status == 'Rejected':
                logger.info(f"  - Action: No change to total_used (rejected)")

            elif old_status in ['Approved', 'Rejected'] and new_status == 'Approved':
                logger.warning(
                    f"  - UNUSUAL: Direct approval without pending status")
                leave_balance.total_used += leave_days
                logger.info(
                    f"  - Action: Added {leave_days} days directly to total_used")

            elif old_status == 'Approved' and new_status == 'Rejected':
                leave_balance.total_used = max(
                    0, leave_balance.total_used - leave_days)
                logger.info(
                    f"  - Action: Removed {leave_days} days from total_used (reversal)")

            # Calculate remaining balance AFTER update
            remaining_after = float(
                leave_balance.total_available - leave_balance.total_used)

            logger.info(f"AFTER UPDATE:")
            logger.info(
                f"  - Total Available: {leave_balance.total_available}")
            logger.info(f"  - Total Used: {leave_balance.total_used}")
            logger.info(f"  - Remaining: {remaining_after}")
            logger.info(
                f"  - Balance Change: {remaining_before} → {remaining_after}")

            db.session.add(leave_balance)
            db.session.commit()

            logger.info(f"✅ Balance update completed for user {leave.user_id}")

        except Exception as e:
            logger.error(f"Error updating leave balance directly: {str(e)}")
            raise

    def get_current_balance_info(self, user_id: str, leave_type_id: str, year: int = None) -> Dict:
        """Get current balance information using analyze_leave_balance"""
        try:
            if year is None:
                year = datetime.now().year

            balance_analysis = analyze_leave_balance(
                user_id, leave_type_id, 0.0, year)

            if balance_analysis.get('error'):
                return {
                    'total_available': 0.0,
                    'total_used': 0.0,
                    'remaining': 0.0,
                    'found': False,
                    'balance_record_id': None
                }

            return {
                'total_available': float(balance_analysis['balance_details']['total_available']),
                'total_used': float(balance_analysis['balance_details']['total_used']),
                'remaining': float(balance_analysis['balance_details']['current_balance']),
                'found': True,
                'balance_record_id': LeaveBalance.query.filter_by(
                    user_id=user_id, leave_type_id=leave_type_id, fiscal_year=year
                ).first().public_id if LeaveBalance.query.filter_by(
                    user_id=user_id, leave_type_id=leave_type_id, fiscal_year=year
                ).first() else None
            }

        except Exception as e:
            logger.error(f"Error getting balance info: {str(e)}")
            return {
                'total_available': 0.0,
                'total_used': 0.0,
                'remaining': 0.0,
                'found': False,
                'balance_record_id': None
            }

    def detect_leave_type(self, reason: str) -> str:
        """Detect appropriate leave type based on the reason provided"""
        reason_lower = reason.lower().strip()

        leave_type_keywords = {
            'sick': ['sick', 'illness', 'medical', 'doctor', 'hospital', 'surgery',
                     'health', 'fever', 'flu', 'cold', 'unwell', 'treatment'],
            'annual': ['vacation', 'holiday', 'annual', 'rest', 'break', 'travel',
                       'family time', 'personal break', 'time off'],
            'personal': ['personal', 'family', 'emergency', 'bereavement', 'funeral',
                         'wedding', 'graduation', 'appointment', 'personal matter'],
            'maternity': ['maternity', 'pregnancy', 'childbirth', 'newborn', 'baby'],
            'paternity': ['paternity', 'father', 'newborn baby', 'child birth'],
            'study': ['study', 'education', 'training', 'course', 'exam', 'learning'],
            'compassionate': ['death', 'funeral', 'bereavement', 'family emergency', 'compassionate']
        }

        for leave_type_name, keywords in leave_type_keywords.items():
            if any(keyword in reason_lower for keyword in keywords):
                leave_type = LeaveType.query.filter(
                    LeaveType.name.ilike(f'%{leave_type_name}%')
                ).first()
                if leave_type:
                    logger.info(
                        f"Detected leave type: {leave_type.name} (ID: {leave_type.id}) for reason: {reason}")
                    return leave_type.id

        default_leave_type = LeaveType.query.filter(
            LeaveType.name.ilike('%annual%')
        ).first()
        if default_leave_type:
            logger.info(
                f"Using default leave type: {default_leave_type.name} (ID: {default_leave_type.id}) for reason: {reason}")
            return default_leave_type.id

        raise ValueError("No leave types found in the system")


class AgenticLeaveSystemSDK:

    def __init__(self):
        self.triage_agent = triage_agent  # Placeholder
        self.context_agent = context_agent  # Placeholder
        self.executor = ThreadPoolExecutor(max_workers=3)
        from app import app
        with app.app_context():
            if not db.session:
                db.session = db.create_scoped_session()
                logger.info("Initialized new database session")

    def validate_and_analyze_leave(self, user_id: str, leave_type_id: str, duration: float,
                                   start_date: str, end_date: str, reason: str, fiscal_year: int = None) -> Dict:
        """Validate and analyze leave request using integrated functions"""
        try:
            from app import app
            with app.app_context():
                validation_result = get_comprehensive_leave_validation(
                    user_id, start_date, end_date)
                analysis_result = get_comprehensive_leave_analysis(
                    user_id, leave_type_id, duration, start_date, end_date, fiscal_year
                )
                precedent_result = get_similar_leave_decisions(
                    reason, duration, user_id, analysis_result['user_info'].get(
                        'team_public_id'), start_date
                )

                ai_decision = {
                    'status': 'Approved',
                    'reason': 'Initial AI evaluation',
                    'confidence': 0.8,
                    'escalate': False,
                    'validation': validation_result,
                    'analysis': analysis_result,
                    'precedents': precedent_result
                }

                if not validation_result.get('overall_valid', True):
                    ai_decision['status'] = 'Rejected'
                    ai_decision[
                        'reason'] = f"Invalid leave request: {validation_result.get('rejection_reasons', [])}"
                    ai_decision['confidence'] = 1.0
                    ai_decision['escalate'] = False
                elif analysis_result['overall_assessment']['recommendation'] in ['rejected', 'conditional']:
                    ai_decision['status'] = 'Escalate'
                    ai_decision['reason'] = f"High risk: {analysis_result['overall_assessment']['blocking_factors']}"
                    ai_decision['confidence'] = 0.6
                    ai_decision['escalate'] = True
                elif precedent_result['historical_approval_rate'] < 40:
                    ai_decision['status'] = 'Escalate'
                    ai_decision['reason'] = f"Low historical approval rate: {precedent_result['historical_approval_rate']}%"
                    ai_decision['confidence'] = 0.5
                    ai_decision['escalate'] = True

                return ai_decision

        except Exception as e:
            logger.error(f"Error in validate_and_analyze_leave: {str(e)}")
            return {
                'status': 'Error',
                'reason': f"Analysis failed: {str(e)}",
                'confidence': 0.0,
                'escalate': True,
                'validation': {},
                'analysis': {},
                'precedents': {}
            }

    def process_leave_decision(self, leave_record, ai_decision: Dict) -> Dict:
        """Process the AI decision and update the leave request status and balance"""
        try:
            from app import app
            with app.app_context():
                if isinstance(leave_record, str):
                    leave = Leave.query.filter_by(
                        public_id=leave_record).first()
                    if not leave:
                        logger.error(
                            f"Leave record with public_id {leave_record} not found")
                        return {"error": f"Leave record {leave_record} not found in database"}
                    logger.info(
                        f"Found Leave object with public_id: {leave.public_id}")

                elif isinstance(leave_record, Leave):
                    leave = leave_record
                    logger.info(
                        f"Using existing Leave object with public_id: {leave.public_id}")

                else:
                    leave = self._find_leave_record(leave_record)
                    if not leave:
                        logger.error(
                            "Could not find Leave record for LeaveRequest")
                        return {"error": "Leave request not found in database"}

                balance_before = self.get_current_balance_info(
                    leave.user_id, leave.leave_type_id, leave.start_date.year
                )
                logger.info(f"🔍 BALANCE BEFORE PROCESSING: {balance_before}")

                old_status = leave.status
                logger.info(
                    f"Processing leave {leave.public_id} - Old status: {old_status}, New status: {ai_decision['status']}")

                leave_days = float(leave.duration) if leave.duration else 0.0
                if leave.is_half_day:
                    leave_days = 0.5 if leave_days == 1.0 else leave_days
                logger.info(f"Leave duration: {leave_days} days")

                if ai_decision['status'] == 'Approved':
                    leave.status = 'Approved'
                    leave.approved_at = datetime.now(timezone.utc)
                    leave.decision_reason = ai_decision['reason']
                    self._update_leave_balance_directly(
                        leave, old_status, 'Approved', leave_days)
                    logger.info(
                        f"✅ Leave request {leave.public_id} approved and balance updated")

                elif ai_decision['status'] in ['Denied', 'Rejected']:
                    leave.status = 'Rejected'
                    leave.decision_reason = ai_decision['reason']
                    self._update_leave_balance_directly(
                        leave, old_status, 'Rejected', leave_days)
                    logger.info(
                        f"❌ Leave request {leave.public_id} rejected and balance updated")

                elif ai_decision['status'] == 'Escalate':
                    leave.status = 'Pending'
                    leave.decision_reason = f"Escalated: {ai_decision['reason']}"
                    logger.info(
                        f"⬆️ Leave request {leave.public_id} escalated for human review")

                db.session.commit()

                balance_after = self.get_current_balance_info(
                    leave.user_id, leave.leave_type_id, leave.start_date.year
                )
                logger.info(f"🔍 BALANCE AFTER PROCESSING: {balance_after}")

                return {
                    "success": True,
                    "leave_id": leave.public_id,
                    "new_status": leave.status,
                    "balance_updated": ai_decision['status'] in ['Approved', 'Rejected'],
                    "balance_before": balance_before,
                    "balance_after": balance_after
                }

        except Exception as e:
            db.session.rollback()
            logger.exception(f"Error processing leave decision: {str(e)}")
            return {
                "error": f"Failed to process decision: {str(e)}",
                "success": False,
                "leave_id": None,
                "new_status": None,
                "balance_updated": False,
                "balance_before": {},
                "balance_after": {}
            }

    def _find_leave_record(self, leave_request) -> Leave:
        """Helper method to find Leave record from LeaveRequest"""
        leave = None

        if hasattr(leave_request, 'public_id') and leave_request.public_id:
            leave = Leave.query.filter_by(
                public_id=leave_request.public_id).first()
            if leave:
                logger.info(
                    f"Found leave by public_id: {leave_request.public_id}")
                return leave

        if hasattr(leave_request, 'user_id'):
            leave = Leave.query.filter_by(
                user_id=leave_request.user_id,
                start_date=leave_request.start_date,
                end_date=leave_request.end_date,
                status='Pending'
            ).order_by(desc(Leave.created_at)).first()
            if leave:
                logger.info(
                    f"Found leave by user_id and dates: {leave.public_id}")
                return leave

        if hasattr(leave_request, 'user_id'):
            leave = Leave.query.filter_by(
                user_id=leave_request.user_id,
                status='Pending'
            ).order_by(desc(Leave.created_at)).first()
            if leave:
                logger.info(
                    f"Found most recent pending leave: {leave.public_id}")
                return leave

        return None

    def _update_leave_balance_directly(self, leave: Leave, old_status: str, new_status: str, leave_days: float):
        """Directly update the leave balance in the LeaveBalance table"""
        try:
            leave_balance = LeaveBalance.query.filter_by(
                user_id=leave.user_id,
                leave_type_id=leave.leave_type_id,
                fiscal_year=leave.start_date.year
            ).first()

            if not leave_balance:
                logger.error(
                    f"No leave balance found for user {leave.user_id}, leave type {leave.leave_type_id}")
                return

            remaining_before = float(
                leave_balance.total_available - leave_balance.total_used)

            logger.info(f"BEFORE UPDATE:")
            logger.info(
                f"  - Total Available: {leave_balance.total_available}")
            logger.info(f"  - Total Used: {leave_balance.total_used}")
            logger.info(f"  - Remaining: {remaining_before}")
            logger.info(f"  - Status Transition: {old_status} → {new_status}")
            logger.info(f"  - Leave Days: {leave_days}")

            if old_status == 'Pending' and new_status == 'Approved':
                leave_balance.total_used += leave_days
                logger.info(
                    f"  - Action: Added {leave_days} days to total_used")

            elif old_status == 'Pending' and new_status == 'Rejected':
                logger.info(f"  - Action: No change to total_used (rejected)")

            elif old_status in ['Approved', 'Rejected'] and new_status == 'Approved':
                logger.warning(
                    f"  - UNUSUAL: Direct approval without pending status")
                leave_balance.total_used += leave_days
                logger.info(
                    f"  - Action: Added {leave_days} days directly to total_used")

            elif old_status == 'Approved' and new_status == 'Rejected':
                leave_balance.total_used = max(
                    0, leave_balance.total_used - leave_days)
                logger.info(
                    f"  - Action: Removed {leave_days} days from total_used (reversal)")

            remaining_after = float(
                leave_balance.total_available - leave_balance.total_used)

            logger.info(f"AFTER UPDATE:")
            logger.info(
                f"  - Total Available: {leave_balance.total_available}")
            logger.info(f"  - Total Used: {leave_balance.total_used}")
            logger.info(f"  - Remaining: {remaining_after}")
            logger.info(
                f"  - Balance Change: {remaining_before} → {remaining_after}")

            db.session.add(leave_balance)
            db.session.commit()

            logger.info(f"✅ Balance update completed for user {leave.user_id}")

        except Exception as e:
            logger.error(f"Error updating leave balance directly: {str(e)}")
            raise

    def get_current_balance_info(self, user_id: str, leave_type_id: str, year: int = None) -> Dict:
        """Get current balance information using analyze_leave_balance"""
        try:
            if year is None:
                year = datetime.now().year

            balance_analysis = analyze_leave_balance(
                user_id, leave_type_id, 0.0, year)

            if balance_analysis.get('error'):
                return {
                    'total_available': 0.0,
                    'total_used': 0.0,
                    'remaining': 0.0,
                    'found': False,
                    'balance_record_id': None
                }

            return {
                'total_available': float(balance_analysis['balance_details']['total_available']),
                'total_used': float(balance_analysis['balance_details']['total_used']),
                'remaining': float(balance_analysis['balance_details']['current_balance']),
                'found': True,
                'balance_record_id': LeaveBalance.query.filter_by(
                    user_id=user_id, leave_type_id=leave_type_id, fiscal_year=year
                ).first().public_id if LeaveBalance.query.filter_by(
                    user_id=user_id, leave_type_id=leave_type_id, fiscal_year=year
                ).first() else None
            }

        except Exception as e:
            logger.error(f"Error getting balance info: {str(e)}")
            return {
                'total_available': 0.0,
                'total_used': 0.0,
                'remaining': 0.0,
                'found': False,
                'balance_record_id': None
            }

    def detect_leave_type(self, reason: str) -> str:
        """Detect appropriate leave type based on the reason provided"""
        reason_lower = reason.lower().strip()

        leave_type_keywords = {
            'sick': ['sick', 'illness', 'medical', 'doctor', 'hospital', 'surgery',
                     'health', 'fever', 'flu', 'cold', 'unwell', 'treatment'],
            'annual': ['vacation', 'holiday', 'annual', 'rest', 'break', 'travel',
                       'family time', 'personal break', 'time off'],
            'personal': ['personal', 'family', 'emergency', 'bereavement', 'funeral',
                         'wedding', 'graduation', 'appointment', 'personal matter'],
            'maternity': ['maternity', 'pregnancy', 'childbirth', 'newborn', 'baby'],
            'paternity': ['paternity', 'father', 'newborn baby', 'child birth'],
            'study': ['study', 'education', 'training', 'course', 'exam', 'learning'],
            'compassionate': ['death', 'funeral', 'bereavement', 'family emergency', 'compassionate']
        }

        for leave_type_name, keywords in leave_type_keywords.items():
            if any(keyword in reason_lower for keyword in keywords):
                leave_type = LeaveType.query.filter(
                    LeaveType.name.ilike(f'%{leave_type_name}%')
                ).first()
                if leave_type:
                    logger.info(
                        f"Detected leave type: {leave_type.name} (ID: {leave_type.id}) for reason: {reason}")
                    return leave_type.id

        default_leave_type = LeaveType.query.filter(
            LeaveType.name.ilike('%annual%')
        ).first()
        if default_leave_type:
            logger.info(
                f"Using default leave type: {default_leave_type.name} (ID: {default_leave_type.id}) for reason: {reason}")
            return default_leave_type.id

        raise ValueError("No leave types found in the system")

    def make_intelligent_decision(self, leave_request, team_members: List[str], additional_context: Dict = None) -> Dict:
        """Process leave request using OpenAI Agents SDK and return AI decision"""
        try:
            logger.info(
                f"🤖 Starting Agents SDK analysis for user {leave_request.user_id}")

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                ai_decision = loop.run_until_complete(
                    self._async_process_request(
                        leave_request, team_members, additional_context)
                )

                if isinstance(ai_decision, str):
                    logger.warning(
                        f"Unexpected output type: {type(ai_decision)}")
                    ai_decision = {
                        'status': 'Escalate',
                        'reason': f"Unexpected AI response format: {ai_decision}",
                        'confidence': 0.5,
                        'escalate': True,
                        'agent_reasoning': f"Output: {ai_decision}",
                        'business_impact': 'unknown',
                        'employee_considerations': 'format_error'
                    }
                elif not isinstance(ai_decision, dict):
                    logger.warning(
                        f"Completely unexpected output type: {type(ai_decision)}")
                    ai_decision = {
                        'status': 'Escalate',
                        'reason': f"AI returned unsupported type: {type(ai_decision)}",
                        'confidence': 0.0,
                        'escalate': True,
                        'agent_reasoning': f"Output: {ai_decision}",
                        'business_impact': 'unknown',
                        'employee_considerations': 'format_error'
                    }

                logger.info("🤖 Agents SDK Decision: %s", ai_decision)
                return ai_decision

            finally:
                loop.close()

        except Exception as e:
            logger.exception(f"Error in Agents SDK processing: {str(e)}")
            return {
                'status': 'Escalate',
                'reason': f'System error during AI analysis: {str(e)}',
                'confidence': 0.0,
                'escalate': True,
                'agent_reasoning': f'SDK Error: {str(e)}',
                'business_impact': 'unknown',
                'employee_considerations': 'system_error',
                'error': str(e)
            }

    def process_leave_request_complete(self, leave_request, team_members: List[str], additional_context: Dict = None) -> Dict:
        """Complete workflow: Get AI decision, save Leave record, then process the decision"""
        try:
            ai_decision = self.make_intelligent_decision(
                leave_request, team_members, additional_context)
            logger.info(f"🤖 AI Decision received: {ai_decision}")
            from app import app
            with app.app_context():
                leave_type_id = getattr(leave_request, 'leave_type_id', None)
                if not leave_type_id:
                    leave_type_id = self.detect_leave_type(
                        leave_request.reason)

                if isinstance(leave_request.start_date, str):
                    start_date = datetime.strptime(
                        leave_request.start_date, '%Y-%m-%d').date()
                else:
                    start_date = leave_request.start_date

                if isinstance(leave_request.end_date, str):
                    end_date = datetime.strptime(
                        leave_request.end_date, '%Y-%m-%d').date()
                else:
                    end_date = leave_request.end_date

                duration = getattr(leave_request, 'duration', None)
                if not duration:
                    duration = float((end_date - start_date).days + 1)
                    if hasattr(leave_request, 'is_half_day') and leave_request.is_half_day:
                        duration = 0.5 if duration == 1.0 else duration

                leave_record = Leave(
                    user_id=leave_request.user_id,
                    leave_type_id=leave_type_id,
                    start_date=start_date,
                    end_date=end_date,
                    reason=leave_request.reason,
                    status='Pending',
                    created_at=datetime.now(timezone.utc),
                    duration=duration,
                    is_half_day=getattr(leave_request, 'is_half_day', False)
                )

                db.session.add(leave_record)
                db.session.commit()
                db.session.refresh(leave_record)
                logger.info(
                    f"✅ Leave saved to database with public_id: {leave_record.public_id}")

                processing_result = self.process_leave_decision(
                    leave_record, ai_decision)

                final_result = {
                    **ai_decision,
                    **processing_result,
                    'leave_id': leave_record.public_id
                }

                logger.info(f"🎯 Final result: {final_result}")
                return final_result

        except Exception as e:
            db.session.rollback()
            logger.exception(
                f"Error in complete leave request processing: {str(e)}")
            return {
                'status': 'Escalate',
                'new_status': 'Pending',
                'reason': f'System error during complete processing: {str(e)}',
                'confidence': 0.0,
                'escalate': True,
                'error': str(e),
                'leave_id': None
            }

    async def _async_process_request(self, leave_request, team_members: List[str], additional_context: Dict = None) -> Dict:
        """Async processing of leave request with comprehensive validation"""
        logger.info("🔍 Running comprehensive leave validation...")

        if isinstance(leave_request.start_date, str):
            start_date_str = leave_request.start_date
            end_date_str = leave_request.end_date
        else:
            start_date_str = leave_request.start_date.strftime('%Y-%m-%d')
            end_date_str = leave_request.end_date.strftime('%Y-%m-%d')
        from app import app
        with app.app_context():
            user = User.query.get(leave_request.user_id)
            if not user:
                logger.error(f"User {leave_request.user_id} not found")
                raise ValueError(f"User {leave_request.user_id} not found")
            team_id = user.primary_team_id

            validation_result = get_comprehensive_leave_validation(
                leave_request.user_id, start_date_str, end_date_str
            )

            if validation_result.get('should_reject', False):
                logger.warning(
                    f"Leave request rejected due to validation failures: {validation_result.get('rejection_reasons', [])}")
                return {
                    'status': 'Rejected',
                    'reason': '; '.join(validation_result.get('rejection_reasons', ['Validation failed'])),
                    'confidence': 1.0,
                    'escalate': False,
                    'agent_reasoning': f"Automated rejection due to validation failures: {validation_result}",
                    'business_impact': 'none',
                    'employee_considerations': 'validation_failure',
                    'precedent_used': 'validation_rules',
                    'recommended_actions': ['Review leave policy and reapply with valid dates'],
                    'validation_details': validation_result
                }

            duration = getattr(leave_request, 'duration', None)
            if not duration:
                duration = float((datetime.strptime(end_date_str, '%Y-%m-%d').date() -
                                 datetime.strptime(start_date_str, '%Y-%m-%d').date()).days + 1)
                if hasattr(leave_request, 'is_half_day') and leave_request.is_half_day:
                    duration = 0.5 if duration == 1.0 else duration

            leave_type_id = getattr(leave_request, 'leave_type_id', None)
            if not leave_type_id:
                leave_type_id = self.detect_leave_type(leave_request.reason)

            analysis_result = self.validate_and_analyze_leave(
                leave_request.user_id, leave_type_id, duration, start_date_str, end_date_str,
                leave_request.reason
            )

            team_analysis = analyze_team_availability(
                team_id or '', start_date_str, end_date_str)

            agent_input = f"""
            NEW LEAVE REQUEST FOR ANALYSIS:

            Employee Details:
            - User ID: {leave_request.user_id}
            - Team ID: {team_id or 'None'}

            Leave Request:
            - Reason: {leave_request.reason}
            - Duration: {duration} days
            - Start Date: {start_date_str}
            - End Date: {end_date_str}
            - Half Day: {getattr(leave_request, 'is_half_day', False)}

            Team Context:
            - Team Size: {len(team_members)} members
            - Team Availability: {team_analysis.get('availability_summary', 'No team data')}

            Validation Results:
            - Overall Valid: {validation_result.get('overall_valid', True)}
            - Working Days: {validation_result.get('summary', {}).get('working_days', 0)}
            - Weekend Days: {validation_result.get('date_validation_result', {}).get('total_weekend_days', 0)}
            - Warnings: {len(validation_result.get('warnings', []))} warnings found
            - Special Approval Required: {validation_result.get('summary', {}).get('requires_special_approval', False)}
            - Manager Approval Required: {validation_result.get('summary', {}).get('requires_manager_approval', False)}

            Analysis Results:
            - Balance Status: {analysis_result['analysis']['balance_status']}
            - Risk Level: {analysis_result['analysis']['risk_level']}
            - Approval Likelihood: {analysis_result['analysis']['overall_assessment']['approval_likelihood']}
            - Historical Approval Rate: {analysis_result['precedents']['historical_approval_rate']}%

            Additional Context: {json.dumps(additional_context or {}, indent=2)}

            Please analyze this request and route to the appropriate agent.
            """

            logger.info("🚀 Sending request to Agents SDK triage system...")

            result = await Runner.run(self.triage_agent, agent_input)
            logger.info(f"Raw agent output: {result.final_output}")

            if isinstance(result.final_output, RoutingDecision):
                routing_decision = result.final_output
            elif isinstance(result.final_output, str):
                try:
                    decision_dict = json.loads(result.final_output)
                    routing_decision = RoutingDecision(**decision_dict)
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Failed to parse JSON output: {str(e)}")
                    return {
                        'status': 'Escalate',
                        'reason': f'Invalid JSON response format: {str(e)}',
                        'confidence': 0.5,
                        'escalate': True,
                        'agent_reasoning': f'Output: {result.final_output}',
                        'business_impact': 'unknown',
                        'employee_considerations': 'format_error',
                        'validation_details': validation_result
                    }
            else:
                logger.warning(
                    f"Unexpected output type: {type(result.final_output)}")
                return {
                    'status': 'Escalate',
                    'reason': f'Unexpected AI response type: {type(result.final_output)}',
                    'confidence': 0.5,
                    'escalate': True,
                    'agent_reasoning': f'Output: {str(result.final_output)}',
                    'business_impact': 'unknown',
                    'employee_considerations': 'format_error',
                    'validation_details': validation_result
                }

            if routing_decision.escalate:
                return {
                    'status': 'Escalate',
                    'reason': routing_decision.reason,
                    'confidence': routing_decision.confidence,
                    'escalate': True,
                    'agent_reasoning': routing_decision.analysis,
                    'business_impact': 'unknown',
                    'employee_considerations': 'triage_escalation',
                    'precedent_used': None,
                    'recommended_actions': [],
                    'validation_details': validation_result
                }

            target_agent = decision_agent if routing_decision.route_to == 'decision_agent' else escalation_agent
            logger.info(f"🚀 Routing to {routing_decision.route_to}...")

            result = await Runner.run(target_agent, agent_input)
            logger.info(
                f"Raw {routing_decision.route_to} output: {result.final_output}")

            if isinstance(result.final_output, LeaveDecision):
                decision = result.final_output
            elif isinstance(result.final_output, str):
                try:
                    decision_dict = json.loads(result.final_output)
                    decision = LeaveDecision(**decision_dict)
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(
                        f"Failed to parse JSON output from {routing_decision.route_to}: {str(e)}")
                    return {
                        'status': 'Escalate',
                        'reason': f'Invalid JSON response format from {routing_decision.route_to}: {str(e)}',
                        'confidence': 0.5,
                        'escalate': True,
                        'agent_reasoning': f'Output: {result.final_output}',
                        'business_impact': 'unknown',
                        'employee_considerations': 'format_error',
                        'validation_details': validation_result
                    }
            else:
                logger.warning(
                    f"Unexpected output type from {routing_decision.route_to}: {type(result.final_output)}")
                return {
                    'status': 'Escalate',
                    'reason': f'Unexpected AI response type from {routing_decision.route_to}: {type(result.final_output)}',
                    'confidence': 0.5,
                    'escalate': True,
                    'agent_reasoning': f'Output: {str(result.final_output)}',
                    'business_impact': 'unknown',
                    'employee_considerations': 'format_error',
                    'validation_details': validation_result
                }

            final_result = {
                'status': decision.status,
                'reason': decision.reason,
                'confidence': decision.confidence,
                'escalate': decision.escalate,
                'agent_reasoning': decision.agent_reasoning,
                'business_impact': decision.business_impact,
                'employee_considerations': decision.employee_considerations,
                'precedent_used': decision.precedent_used,
                'recommended_actions': decision.recommended_actions,
                'validation_details': validation_result
            }

            if validation_result.get('warnings'):
                final_result['validation_warnings'] = validation_result['warnings']

            return final_result


class LeaveBalance(db.Model):
    __tablename__ = 'leave_balance'

    id = db.Column(db.String(20), primary_key=True)
    public_id = db.Column(db.String(36), unique=True, nullable=False)
    user_id = db.Column(db.String(36), db.ForeignKey(
        'user.id'), nullable=False)
    department_id = db.Column(db.String(36), db.ForeignKey(
        'department.id'), nullable=False)
    organization_id = db.Column(db.String(36), db.ForeignKey(
        'organization.id'), nullable=False)
    leave_type_id = db.Column(db.String(36), db.ForeignKey(
        'leave_type.id'), nullable=False)
    fiscal_year = db.Column(db.Integer, nullable=False)
    total_available = db.Column(db.Numeric(5, 2), default=0.0)
    total_used = db.Column(db.Numeric(5, 2), default=0.0)
    annual_allocation = db.Column(db.Numeric(5, 2), default=0.0)
    carried_forward = db.Column(db.Numeric(5, 2), default=0.0)
    carry_forward_expiry = db.Column(
        db.Date, default=lambda: date(datetime.now().year, 12, 31))
    accrued_balance = db.Column(db.Numeric(5, 2), default=0.0)
    last_accrual_date = db.Column(db.Date)
    created_at = db.Column(
        db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(
        db.DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    user = db.relationship('User', back_populates='leave_balances')
    department = db.relationship('Department', back_populates='leave_balances')
    organization = db.relationship(
        'Organization', back_populates='leave_balances')
    leave_type = db.relationship('LeaveType', back_populates='leave_balances')

    __table_args__ = (
        db.UniqueConstraint('user_id', 'leave_type_id', 'fiscal_year',
                            name='unique_balance_per_user_type_year'),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.public_id:
            self.public_id = str(uuid.uuid4())

    @staticmethod
    def expire_carry_forward_balances():
        today = datetime.now().date()
        expired_balances = LeaveBalance.query.filter(
            LeaveBalance.carry_forward_expiry <= today
        ).all()

        for balance in expired_balances:
            balance.carry_forward = 0
            balance.carry_forward_expiry = None
            db.session.add(balance)
        db.session.commit()


class Leave(db.Model):
    __tablename__ = 'leave'

    id = db.Column(db.String(20), primary_key=True)
    public_id = db.Column(db.String(36), unique=True, nullable=False)
    user_id = db.Column(db.String(36), db.ForeignKey(
        'user.id', ondelete='CASCADE'), nullable=False)
    leave_type_id = db.Column(db.String(36), db.ForeignKey(
        'leave_type.id'), nullable=False)
    department_id = db.Column(db.String(36), db.ForeignKey(
        'department.id'), nullable=False)
    organization_id = db.Column(db.String(36), db.ForeignKey(
        'organization.id'), nullable=False)
    start_date = db.Column(db.Date, nullable=False)
    end_date = db.Column(db.Date, nullable=False)
    is_half_day = db.Column(db.Boolean, default=False)
    half_day_type = db.Column(db.String(10))
    duration = db.Column(db.Numeric(5, 2))
    reason = db.Column(db.String(500))
    status = db.Column(db.String(20), default='Pending')
    attachment_url = db.Column(db.String(500))
    emergency_contact = db.Column(db.String(200))
    work_handover_to = db.Column(db.String(36), db.ForeignKey('user.id'))
    cancellation_reason = db.Column(db.String(500))
    cancelled_at = db.Column(db.DateTime)
    reviewed_by = db.Column(db.String(36), db.ForeignKey('user.id'))
    reviewed_at = db.Column(db.DateTime)
    review_comments = db.Column(db.String(500))
    created_at = db.Column(
        db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(
        db.DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    user = db.relationship(
        'User', back_populates='leaves', foreign_keys=[user_id])
    leave_type = db.relationship('LeaveType', back_populates='leaves')
    department = db.relationship('Department', back_populates='leaves')
    organization = db.relationship('Organization', back_populates='leaves')
    handover_user = db.relationship(
        'User', back_populates='handover_leaves', foreign_keys=[work_handover_to])
    reviewer = db.relationship(
        'User', back_populates='reviewed_leaves', foreign_keys=[reviewed_by])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.public_id:
            self.public_id = str(uuid.uuid4())


@event.listens_for(Leave, 'before_insert')
def set_leave_id(mapper, connection, target):
    if not target.id:
        existing_ids = {row[0] for row in connection.execute(
            select(Leave.id)).fetchall()}
        target.id = generate_id("L-", 12, existing_ids)

        class LeaveApplicationForm(FlaskForm):
    leave_type = SelectField('Leave Type', validators=[DataRequired()])
    reason = TextAreaField('Reason', validators=[DataRequired()])
    start_date = DateField('Start Date', validators=[DataRequired()])
    end_date = DateField('End Date', validators=[DataRequired()])
    is_half_day = BooleanField('Half Day')
    emergency_contact = StringField(
        'Emergency Contact', validators=[DataRequired()])
    work_handover_to = SelectField('Work Handover To',
                                   validators=[Optional()],
                                   render_kw={"class": "form-select"})
    attachments = FileField('Attachments', validators=[Optional()])

    def __init__(self, *args, **kwargs):
        super(LeaveApplicationForm, self).__init__(*args, **kwargs)
        self.current_user = kwargs.get('current_user')
        # Populate the leave type choices from the database
        self.leave_type.choices = [(leave_type.name, leave_type.name)
                                   for leave_type in LeaveType.query.all()]

        # Dynamically populate the 'work_handover_to' field with colleagues in the same department
        if 'colleagues' in kwargs:
            self.work_handover_to.choices = [('', 'Select Colleague')] + [
                (str(c.id), f"{c.email} ({c.id})") for c in kwargs['colleagues']
            ]

    def validate_start_date(self, field):
        """Check if start_date is not before today and not on a weekend or holiday"""
        today = date.today()
        if field.data < today:
            raise ValidationError('Start date cannot be before today.')
        if field.data.weekday() in [5, 6]:
            raise ValidationError(
                'Start date cannot be on a weekend (Saturday or Sunday).')

        # Check if start_date is a holiday
        holiday = Holiday.query.filter(
            Holiday.organization_id == self.current_user.primary_organization_id,
            Holiday.date == field.data
        )
        if self.current_user.primary_department_id:
            holiday = holiday.filter(
                Holiday.department_id == self.current_user.primary_department_id)
        holiday = holiday.first()
        if holiday:
            raise ValidationError(
                f'Start date falls on a holiday: {holiday.name}.')

    def validate_end_date(self, field):
        """Check if end_date is not before start_date and not on a weekend or holiday"""
        if self.start_date.data and field.data < self.start_date.data:
            raise ValidationError('End date cannot be before start date.')
        if field.data.weekday() in [5, 6]:
            raise ValidationError(
                'End date cannot be on a weekend (Saturday or Sunday).')

        # Check if end_date is a holiday
        holiday = Holiday.query.filter(
            Holiday.organization_id == self.current_user.primary_organization_id,
            Holiday.date == field.data
        )
        if self.current_user.primary_department_id:
            holiday = holiday.filter(
                Holiday.department_id == self.current_user.primary_department_id)
        holiday = holiday.first()
        if holiday:
            raise ValidationError(
                f'End date falls on a holiday: {holiday.name}.')


@app.route('/apply_leave', methods=['GET', 'POST'])
@role_required(['user'])
def apply_leave():
    email = session.get('email')
    current_user = User.query.filter_by(email=email).first()
    if not current_user:
        flash('User not found', 'danger')
        return redirect(url_for('login'))
    print(
        f"Step 1: Current User - ID: {current_user.id}, Email: {current_user.email}, Org: {current_user.primary_organization_id}, Dept: {current_user.primary_department_id}")

    if not current_user.primary_department_id or not current_user.primary_organization_id:
        flash('User must have a primary department and organization assigned to apply for leave', 'danger')
        return redirect(url_for('dashboard'))

    working_year = WorkingYearConfig.query.filter_by(
        organization_id=current_user.primary_organization_id,
        department_id=current_user.primary_department_id,
        year=datetime.now().year
    ).first() or WorkingYearConfig.query.filter_by(
        organization_id=current_user.primary_organization_id,
        department_id=None,
        year=datetime.now().year
    ).first()
    if not working_year:
        flash('Working year configuration not found for your organization', 'danger')
        return redirect(url_for('dashboard'))
    print(
        f"Step 2: Working Year - ID: {working_year.id}, Hours/Day: {working_year.standard_work_hours_per_day}, Total Hours: {working_year.total_working_hours}")

    hours_per_day = working_year.standard_work_hours_per_day

    leave_balances = LeaveBalance.query.filter_by(
        user_id=current_user.id,
        fiscal_year=str(datetime.now().year)  # Ensure string
    ).all()
    print(
        f"Step 3: Existing Leave Balances - Count: {len(leave_balances)}, Balances: {[{lb.leave_type_id: LeaveType.query.get(lb.leave_type_id).name} for lb in leave_balances]}")

    today = datetime.now(timezone.utc).date()
    service_years = Decimal(
        (today - current_user.join_date).days) / Decimal('365.25')
    print(f"Step 4: Today: {today}, Service Years: {service_years}")

    # Initialize leave balances if they don't exist (same as before)
    if not leave_balances:
        leave_types = LeaveType.query.filter_by(
            organization_id=current_user.primary_organization_id,
            department_id=current_user.primary_department_id
        ).all()
        if not leave_types:
            flash('No leave types defined for your department. Contact admin.', 'danger')
            return redirect(url_for('dashboard'))
        print(
            f"Step 5: Leave Types - Count: {len(leave_types)}, Names: {[lt.name for lt in leave_types]}")

        for leave_type in leave_types:
            existing = LeaveBalance.query.filter_by(
                user_id=current_user.id,
                leave_type_id=leave_type.id,
                fiscal_year=str(datetime.now().year)
            ).first()
            print(
                f"Step 6: Checking Leave Type - {leave_type.name}, Existing Balance: {existing is not None}")

            if not existing:
                config = LeaveAccrualConfiguration.query.filter(
                    LeaveAccrualConfiguration.leave_type_id == leave_type.id,
                    LeaveAccrualConfiguration.department_id == current_user.primary_department_id,
                    LeaveAccrualConfiguration.min_years_service <= service_years,
                    LeaveAccrualConfiguration.max_years_service >= service_years
                ).first()
                print(
                    f"Step 7: Config for {leave_type.name} - Found: {config is not None}, Config ID: {config.id if config else 'None'}")

                if not config:
                    print(
                        f"Step 7.1: No department-specific config for {leave_type.name}, skipping")
                    continue

                annual_hours = config.annual_hours
                accrual_rate = config.accrual_rate
                accrued_hours = accrual_rate
                print(
                    f"Step 8: Creating Balance for {leave_type.name} - Annual Hours: {annual_hours}, Accrual Rate: {accrual_rate}, Accrued Hours: {accrued_hours}")

                leave_balance = LeaveBalance(
                    user_id=current_user.id,
                    leave_type_id=leave_type.id,
                    fiscal_year=str(datetime.now().year),
                    department_id=current_user.primary_department_id,
                    organization_id=current_user.primary_organization_id,
                    total_available=accrued_hours,
                    annual_allocation=annual_hours,
                    carried_forward=Decimal('0.0'),
                    accrued_balance=accrued_hours,
                    total_used=Decimal('0.0')
                )
                db.session.add(leave_balance)

        try:
            db.session.commit()
            print("Step 9.1: Leave balances committed successfully")
        except Exception as e:
            db.session.rollback()
            print(
                f"Step 9.2: Failed to commit leave balances - Error: {str(e)}")
            flash(f'Error creating leave balances: {str(e)}', 'danger')
            return redirect(url_for('dashboard'))

        leave_balances = LeaveBalance.query.filter_by(
            user_id=current_user.id,
            fiscal_year=str(datetime.now().year)
        ).all()
        print(
            f"Step 9.3: New Leave Balances Created - Count: {len(leave_balances)}")

    leave_summary = current_user.get_total_leave_summary()
    colleagues = User.query.filter(
        User.primary_department_id == current_user.primary_department_id,
        User.id != current_user.id
    ).all()
    print(f"Step 10: Colleagues - Count: {len(colleagues)}")

    pending_leaves_count = Leave.query.filter_by(
        user_id=current_user.id,
        status='Pending'
    ).count()
    print(f"Step 11: Pending Leaves Count: {pending_leaves_count}")

    form = LeaveApplicationForm(
        colleagues=colleagues, current_user=current_user)
    form.work_handover_to.choices = [('', 'Select Colleague')] + [
        (str(colleague.id), f"{colleague.email}") for colleague in colleagues
    ]
    leave_types = LeaveType.query.filter_by(
        organization_id=current_user.primary_organization_id,
        department_id=current_user.primary_department_id
    ).all()
    form.leave_type.choices = [(lt.id, lt.name) for lt in leave_types]
    leave_types_serializable = [
        {'name': lt.name, 'requires_attachment': bool(lt.requires_attachment)}
        for lt in leave_types
    ]
    print(f"Step 12: Serializable Leave Types - {leave_types_serializable}")

    print(
        f"Step 12.1: CSRF Token in Session - {session.get('_csrf_token', 'Not found')}")

    if request.method == 'POST':
        try:
            token = request.form.get(
                'csrf_token') or request.headers.get('X-CSRF-Token')
            print("Step 12.2: CSRF token received: ", token)
            validate_csrf(token)
        except CSRFError as e:
            print(f"Step 12.3: CSRF validation failed: {str(e)}")
            if request.is_xhr or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'error': 'Invalid CSRF token'}), 400
            else:
                flash('Invalid CSRF token', 'danger')
                return redirect(url_for('apply_leave'))

        print(f"Step 12.4: Form data submitted: {request.form}")
        if form.validate_on_submit():
            print("Step 12.5: Form validated successfully")
            if form.is_half_day.data and form.start_date.data != form.end_date.data:
                form.is_half_day.errors.append(
                    "Half day leave must be for a single day only.")
                return render_template('apply_leave.html',
                                       form=form,
                                       leave_balances=leave_balances,
                                       pending_leaves_count=pending_leaves_count,
                                       leave_summary=leave_summary,
                                       hours_per_day=hours_per_day,
                                       leave_types=leave_types_serializable)

            try:
                leave_type_entry = LeaveType.query.filter_by(
                    id=form.leave_type.data,
                    organization_id=current_user.primary_organization_id,
                    department_id=current_user.primary_department_id
                ).first()
                if not leave_type_entry:
                    flash('Invalid leave type selected', 'danger')
                    return redirect(url_for('apply_leave'))
                print(
                    f"Step 13: Selected Leave Type - {leave_type_entry.name}, ID: {leave_type_entry.id}")

                start_date = form.start_date.data
                end_date = form.end_date.data
                is_half_day = form.is_half_day.data
                print(
                    f"Step 13.1: Start Date: {start_date}, End Date: {end_date}, Is Half Day: {is_half_day}")

                # Calculate working days (same logic as before)
                holidays = Holiday.query.filter(
                    Holiday.organization_id == current_user.primary_organization_id,
                    Holiday.date.between(start_date, end_date)
                )
                if current_user.primary_department_id:
                    holidays = holidays.filter(
                        Holiday.department_id == current_user.primary_department_id)
                holidays = holidays.all()
                holiday_dates = {h.date for h in holidays}
                print(f"Step 13.2: Holiday dates: {holiday_dates}")

                days = 0
                current_date = start_date
                print(
                    f"Step 13.3: Checking dates from {start_date} to {end_date}")
                while current_date <= end_date:
                    weekday = current_date.weekday()
                    is_holiday = current_date in holiday_dates
                    is_counted = weekday < 5 and not is_holiday
                    print(
                        f"Date: {current_date}, Weekday: {weekday}, Is Weekend: {weekday >= 5}, Is Holiday: {is_holiday}, Counted: {is_counted}")
                    if is_counted:
                        days += 1
                    current_date += timedelta(days=1)

                if days == 0:
                    flash(
                        'Selected leave period contains only weekends or holidays. Please choose valid working days.', 'danger')
                    return redirect(url_for('apply_leave'))

                duration_hours = Decimal(str(days)) * \
                    Decimal(str(hours_per_day))
                if is_half_day:
                    duration_hours = Decimal(
                        str(hours_per_day)) / Decimal('2.0')
                print(
                    f"Step 14: Leave Duration - Start: {start_date}, End: {end_date}, Half Day: {is_half_day}, Working Days: {days}, Duration Hours: {duration_hours}")

                # Handle attachment upload
                attachment_url = None
                if form.attachments.data:
                    file = form.attachments.data
                    filename = secure_filename(file.filename)
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    unique_filename = f"{current_user.id}_{timestamp}_{filename}"
                    upload_folder = os.path.join(app.root_path, 'Uploads')
                    if not os.path.exists(upload_folder):
                        os.makedirs(upload_folder)
                    filepath = os.path.join(upload_folder, unique_filename)
                    file.save(filepath)
                    attachment_url = filepath
                print(f"Step 17: Attachment - URL: {attachment_url}")

                # 🤖 AGENTIC PROCESSING STARTS HERE
                logger.info(
                    f"🚀 New Agentic SDK leave request from user {current_user.id}")

                # Create leave request object for agentic processing
                class LeaveRequestObject:
                    def __init__(self):
                        self.user_id = current_user.id
                        self.leave_type_id = leave_type_entry.id
                        self.start_date = start_date
                        self.end_date = end_date
                        self.duration = float(duration_hours)
                        self.reason = form.reason.data
                        self.is_half_day = is_half_day
                        self.emergency_contact = form.emergency_contact.data
                        self.work_handover_to = form.work_handover_to.data or None
                        self.attachment_url = attachment_url
                        self.department_id = current_user.primary_department_id
                        self.organization_id = current_user.primary_organization_id

                leave_request = LeaveRequestObject()

                # Get team members for context
                team_members = [colleague.id for colleague in colleagues]

                # Prepare additional context
                additional_context = {
                    'request_source': 'web_form',
                    'user_role': getattr(current_user, 'role', 'user'),
                    'department_name': current_user.primary_department.name if current_user.primary_department else 'Unknown',
                    'organization_name': current_user.primary_organization.name if current_user.primary_organization else 'Unknown',
                    'service_years': float(service_years),
                    'working_days': days,
                    'hours_per_day': float(hours_per_day)
                }

                # Use the agentic system
                agent_system = AgenticLeaveSystemSDK()
                result = agent_system.process_leave_request_complete(
                    leave_request,
                    team_members,
                    additional_context
                )

                logger.info(f"🤖 Complete agentic processing result: {result}")

                if result.get('error'):
                    flash(
                        f'Error processing leave request: {result["error"]}', 'danger')
                    return redirect(url_for('apply_leave'))

                # Enhanced flash messages with Agentic SDK insights
                confidence_text = f" (AI Confidence: {result.get('confidence', 0):.0%})"
                flash_message = f"🤖 Leave {result['new_status']}! {result.get('reason', '')}{confidence_text}"

                # Add balance information
                if result.get('balance_after'):
                    balance_info = result['balance_after']
                    balance_text = f" | Remaining balance: {balance_info.get('remaining', 'N/A')} days"
                    flash_message += balance_text

                # Add business impact info if available
                if result.get('business_impact') and result['business_impact'] != 'unknown':
                    flash_message += f" | Business Impact: {result['business_impact']}"

                flash_type = 'success' if result['new_status'] == 'Approved' else 'warning' if result[
                    'new_status'] == 'Pending' else 'info'
                flash(flash_message, flash_type)

                # Show reasoning in a separate message for transparency
                if result.get('agent_reasoning'):
                    reasoning_msg = f"🧠 AI Reasoning: {result['agent_reasoning'][:200]}..."
                    flash(reasoning_msg, 'info')

                # Show balance change details
                if result.get('balance_before') and result.get('balance_after'):
                    before = result['balance_before']
                    after = result['balance_after']
                    balance_change_msg = f"📊 Balance Update: Pending {before.get('pending', 0)} → {after.get('pending', 0)}, Remaining {before.get('remaining', 0)} → {after.get('remaining', 0)}"
                    flash(balance_change_msg, 'info')

                # Send email notification (same as before)
                sender_email = "parvez@assiduusinc.com"
                recipient = current_user.manager if current_user.manager else User.query.filter_by(
                    role='admin').first()
                if recipient:
                    recipient_role = "manager" if current_user.manager else "admin"
                    dept_name = current_user.primary_department.name if current_user.primary_department else "Unknown Department"
                    handover_user = User.query.get(
                        leave_request.work_handover_to) if leave_request.work_handover_to else None
                    handover_email = handover_user.email if handover_user else "Not assigned"

                    with open('templates/applyleaveemail.html', 'r') as file:
                        email_template = file.read()

                    html_body = render_template_string(
                        email_template,
                        recipient_role=recipient_role,
                        user_email=current_user.email,
                        dept_name=dept_name,
                        leave_type=leave_type_entry.name,
                        duration_hours=duration_hours,
                        is_half_day=is_half_day,
                        days=days,
                        start_date=start_date,
                        end_date=end_date,
                        reason=form.reason.data or 'Not specified',
                        handover_email=handover_email,
                        app_url="http://127.0.0.1:5000/leave_list"
                    )

                    text_body = f"""
                    New Leave Request (AI Processed: {result['new_status']})
                    Hi {recipient_role},
                    A user, {current_user.email}, from {dept_name} has requested a leave:
                    - Leave Type: {leave_type_entry.name}
                    - Duration: {duration_hours} hours ({'Half Day' if is_half_day else f'{days} Day(s)'})
                    - From: {start_date}
                    - To: {end_date}
                    - Reason: {form.reason.data or 'Not specified'}
                    - Work Handover: {handover_email}
                    - AI Decision: {result['new_status']} (Confidence: {result.get('confidence', 0):.0%})
                    - AI Reasoning: {result.get('reason', 'N/A')}
                    Please review: http://127.0.0.1:5000/leave_list
                    Regards,
                    AI-Powered Leave Management System
                    """

                    email_result = send_email(
                        sender_email, recipient.email, f"🤖 AI-Processed Leave Request: {result['new_status']}", html_body, text_body)

                    if not email_result["success"]:
                        flash(
                            f'Leave processed successfully but notification failed: {email_result["error"]}', 'warning')

                return redirect(url_for('leave_list'))

            except Exception as e:
                db.session.rollback()
                print(f"Step 19: Error submitting leave - {str(e)}")
                logger.exception(f"Error processing leave request: {str(e)}")
                flash(
                    f'Error processing your leave request: {str(e)}. Please contact HR.', 'danger')
                return redirect(url_for('apply_leave'))
        else:
            print(
                f"Step 12.6: Form validation failed with errors: {form.errors}")
            flash('Please correct the errors in the form.', 'danger')
            return render_template('apply_leave.html',
                                   form=form,
                                   leave_balances=leave_balances,
                                   pending_leaves_count=pending_leaves_count,
                                   leave_summary=leave_summary,
                                   hours_per_day=hours_per_day,
                                   leave_types=leave_types_serializable)
    else:
        return render_template('apply_leave.html',
                               form=form,
                               leave_balances=leave_balances,
                               pending_leaves_count=pending_leaves_count,
                               leave_summary=leave_summary,
                               hours_per_day=hours_per_day,
                               leave_types=leave_types_serializable)
