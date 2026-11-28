| Feature | Group | Type | Justification |
|---|---|---|---|
| day_of_week | calendar | known-in-advance | derived from the scheduled date/time only |
| day_of_month | calendar | known-in-advance | derived from the scheduled date/time only |
| is_weekend | calendar | known-in-advance | derived from the scheduled date/time only |
| dep_hour | calendar | known-in-advance | derived from the scheduled date/time only |
| dep_minute_of_day | calendar | known-in-advance | derived from the scheduled date/time only |
| hour_sin | calendar | known-in-advance | derived from the scheduled date/time only |
| hour_cos | calendar | known-in-advance | derived from the scheduled date/time only |
| days_to_holiday | calendar | known-in-advance | derived from the scheduled date/time only |
| is_holiday | calendar | known-in-advance | derived from the scheduled date/time only |
| OP_UNIQUE_CARRIER | schedule_static | known-in-advance / static | timetable and identity of carrier, airports, route |
| MKT_UNIQUE_CARRIER | schedule_static | known-in-advance / static | timetable and identity of carrier, airports, route |
| is_codeshare | schedule_static | known-in-advance / static | timetable and identity of carrier, airports, route |
| ORIGIN_CITY_MARKET_ID | schedule_static | known-in-advance / static | timetable and identity of carrier, airports, route |
| DEST_CITY_MARKET_ID | schedule_static | known-in-advance / static | timetable and identity of carrier, airports, route |
| ORIGIN_STATE_ABR | schedule_static | known-in-advance / static | timetable and identity of carrier, airports, route |
| DEST_STATE_ABR | schedule_static | known-in-advance / static | timetable and identity of carrier, airports, route |
| DISTANCE | schedule_static | known-in-advance / static | timetable and identity of carrier, airports, route |
| dep_period | schedule_static | known-in-advance / static | timetable and identity of carrier, airports, route |
| tail_leg_number | aircraft_schedule | known-in-advance | aircraft rotation is in the published schedule |
| tail_legs_today | aircraft_schedule | known-in-advance | aircraft rotation is in the published schedule |
| has_prev_leg | aircraft_schedule | known-in-advance | aircraft rotation is in the published schedule |
| sched_gap_prev_leg | aircraft_schedule | known-in-advance | aircraft rotation is in the published schedule |
| planned_turn_slack | aircraft_schedule | known-in-advance | aircraft rotation is in the published schedule |
| prev_leg_departed_by_T | aircraft_observed | observed-by-T | uses earlier legs only if they had departed by T (else 'still on ground') |
| prev_leg_delay_obs | aircraft_observed | observed-by-T | uses earlier legs only if they had departed by T (else 'still on ground') |
| prev_leg_delay_lb | aircraft_observed | observed-by-T | uses earlier legs only if they had departed by T (else 'still on ground') |
| prev_leg_overdue | aircraft_observed | observed-by-T | uses earlier legs only if they had departed by T (else 'still on ground') |
| propagated_delay_est | aircraft_observed | observed-by-T | uses earlier legs only if they had departed by T (else 'still on ground') |
| last_observed_leg_delay | aircraft_observed | observed-by-T | uses earlier legs only if they had departed by T (else 'still on ground') |
| legs_since_observed | aircraft_observed | observed-by-T | uses earlier legs only if they had departed by T (else 'still on ground') |
| tail_late_legs_last3_obs | aircraft_observed | observed-by-T | uses earlier legs only if they had departed by T (else 'still on ground') |
| origin_hourly_flights | congestion_scheduled | known-in-advance | counts of scheduled flights, no outcomes |
| origin_daily_flights | congestion_scheduled | known-in-advance | counts of scheduled flights, no outcomes |
| dest_daily_flights | congestion_scheduled | known-in-advance | counts of scheduled flights, no outcomes |
| carrier_origin_daily_flights | congestion_scheduled | known-in-advance | counts of scheduled flights, no outcomes |
| origin_hour_share | congestion_scheduled | known-in-advance | counts of scheduled flights, no outcomes |
| origin_recent_n_dep | congestion_observed | observed-by-T | other flights whose actual departure was in (T-2h, T] |
| origin_recent_mean_delay | congestion_observed | observed-by-T | other flights whose actual departure was in (T-2h, T] |
| origin_recent_late_rate | congestion_observed | observed-by-T | other flights whose actual departure was in (T-2h, T] |
| dest_recent_n_dep | congestion_observed | observed-by-T | other flights whose actual departure was in (T-2h, T] |
| dest_recent_mean_delay | congestion_observed | observed-by-T | other flights whose actual departure was in (T-2h, T] |
| dest_recent_late_rate | congestion_observed | observed-by-T | other flights whose actual departure was in (T-2h, T] |
| carrier_recent_mean_delay | congestion_observed | observed-by-T | other flights whose actual departure was in (T-2h, T] |
| carrier_recent_late_rate | congestion_observed | observed-by-T | other flights whose actual departure was in (T-2h, T] |
| national_recent_mean_delay | congestion_observed | observed-by-T | other flights whose actual departure was in (T-2h, T] |
| national_recent_late_rate | congestion_observed | observed-by-T | other flights whose actual departure was in (T-2h, T] |
| origin_backlog_at_T | congestion_observed | observed-by-T | other flights whose actual departure was in (T-2h, T] |
| origin_hist_late_rate | historical_rates | observed-by-T | LATE rate on strictly earlier days, DEV days only (no test data) |
| dest_hist_late_rate | historical_rates | observed-by-T | LATE rate on strictly earlier days, DEV days only (no test data) |
| carrier_hist_late_rate | historical_rates | observed-by-T | LATE rate on strictly earlier days, DEV days only (no test data) |
| route_hist_late_rate | historical_rates | observed-by-T | LATE rate on strictly earlier days, DEV days only (no test data) |
| route_hist_n | historical_rates | observed-by-T | LATE rate on strictly earlier days, DEV days only (no test data) |
| tail_hist_late_rate | historical_rates | observed-by-T | LATE rate on strictly earlier days, DEV days only (no test data) |
| tail_hist_n | historical_rates | observed-by-T | LATE rate on strictly earlier days, DEV days only (no test data) |
| hour_hist_late_rate | historical_rates | observed-by-T | LATE rate on strictly earlier days, DEV days only (no test data) |
| DEP_DELAY | label source | outcome-like | used ONLY to make LATE; never a feature |
| LATE | target | outcome | DEP_DELAY >= 15 min |
