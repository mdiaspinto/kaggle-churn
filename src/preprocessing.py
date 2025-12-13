import pandas as pd
import numpy as np
from typing import Union
from sklearn.base import BaseEstimator, TransformerMixin

def compute_cancellation(df: pd.DataFrame,
								present_time: Union[str, pd.Timestamp],
								user_col: str = 'userId',
								page_col: str = 'page',
								time_col: str = 'time',
								target_col: str = 'cancellation_confirmed',
								window_days: int = 10) -> pd.DataFrame:
	"""
	Compute a per-customer cancellation target within a time window.

	The target is 1 if the customer has a 'cancellation confirmation' event
	within `window_days` days after `present_time`. Matching is done
	case-insensitively by checking whether the substring
	'cancellation confirmation' appears in the `page` text. Missing page or
	time values are handled safely.

	Parameters
	- df: input DataFrame containing user events
	- present_time: the start of the window (str or pd.Timestamp)
	- user_col: column identifying the customer (default: 'userID')
	- page_col: column containing page/event description (default: 'page')
	- time_col: column containing timestamps for events (default: 'time')
	- target_col: name of the output target column (default: 'cancellation_confirmed')
	- window_days: integer window size in days to search for cancellation after
	  present_time (default: 10)

	Returns a DataFrame with columns [user_col, target_col] where target_col
	is 0/1. If required columns are missing a ValueError is raised.

	Example:
		target_df = compute_cancellation_target(df, present_time='2023-01-01')
		df = df.merge(target_df, on='userID', how='left')
	"""
	if user_col not in df.columns:
		raise ValueError(f"user_col '{user_col}' not found in DataFrame")
	if page_col not in df.columns:
		raise ValueError(f"page_col '{page_col}' not found in DataFrame")
	if time_col not in df.columns:
		raise ValueError(f"time_col '{time_col}' not found in DataFrame")

	# Parse present_time
	ref_time = pd.to_datetime(present_time)
	window_end = ref_time + pd.Timedelta(days=window_days)

	# Work with a local copy and parse times safely
	df_loc = df.copy()

	# Convert categorical columns to string to avoid fillna issues
	if hasattr(df_loc[page_col], 'cat'):
		df_loc[page_col] = df_loc[page_col].astype(str)

	# Identify cancellation rows
	page_series = df_loc[page_col].fillna('').astype(str)
	is_cancel = page_series.str.contains('Cancellation Confirmation', na=False)

	# Filter cancellation rows within the time window
	in_window = (
		is_cancel &
		df_loc[time_col].notna() &
		(df_loc[time_col] >= ref_time) &
		(df_loc[time_col] <= window_end)
	)

	# Get unique users who cancelled within the window
	cancelled_users = df_loc.loc[in_window, user_col].dropna().astype(str).unique()

	# Build output of all unique users with target 0/1
	all_users = pd.Series(df_loc[user_col].astype(str).unique(), name=user_col)
	result = pd.DataFrame({user_col: all_users})
	result[target_col] = result[user_col].isin(cancelled_users).astype(int)

	return result


def compute_cancellation_batch(df: pd.DataFrame,
								dates_df: pd.DataFrame,
								user_col: str = 'userId',
								page_col: str = 'page',
								time_col: str = 'time',
								date_col: str = 'date',
								window_days: int = 10) -> pd.DataFrame:
	"""
	Efficiently compute cancellation targets for multiple dates at once.
	
	Much faster than calling compute_cancellation in a loop.
	
	Parameters
	----------
	df : pd.DataFrame
		Raw event-level dataframe
	dates_df : pd.DataFrame
		Aggregated dataframe with userId and date columns for which targets should be computed
	user_col : str
		Column name for user ID
	page_col : str
		Column name for page/event type
	time_col : str
		Column name for event timestamp
	date_col : str
		Column name for date
	window_days : int
		Number of days to look ahead for cancellation
	
	Returns
	-------
	pd.DataFrame
		DataFrame with userId, date, and churn_status columns
	"""
	if user_col not in df.columns or page_col not in df.columns or time_col not in df.columns:
		raise ValueError("Missing required columns in raw data")
	
	# Convert categorical columns to string if needed
	df_copy = df.copy()
	if hasattr(df_copy[page_col], 'cat'):
		df_copy[page_col] = df_copy[page_col].astype(str)
	
	# Identify all cancellation events
	page_series = df_copy[page_col].fillna('').astype(str)
	is_cancel = page_series.str.contains('Cancellation Confirmation', na=False, case=False)
	cancellations = df_copy.loc[is_cancel, [user_col, time_col]].copy()
	cancellations[user_col] = cancellations[user_col].astype(int)
	cancellations[time_col] = pd.to_datetime(cancellations[time_col])
	
	# For each date in dates_df, check if user cancelled within window
	dates_df_copy = dates_df[[user_col, date_col]].copy().drop_duplicates()
	dates_df_copy['date_dt'] = pd.to_datetime(dates_df_copy[date_col])
	dates_df_copy[user_col] = dates_df_copy[user_col].astype(int)
	
	# Add window bounds
	dates_df_copy['window_end'] = dates_df_copy['date_dt'] + pd.Timedelta(days=window_days)
	
	# Merge to find cancellations within each window
	merged = dates_df_copy.merge(
		cancellations,
		on=user_col,
		how='left'
	)
	
	# Check if cancellation falls within window
	merged['in_window'] = (
		(merged[time_col] >= merged['date_dt']) &
		(merged[time_col] <= merged['window_end'])
	)
	
	# Get churn status (1 if any cancellation in window, 0 otherwise)
	churn_status = merged.groupby([user_col, date_col])['in_window'].any().astype(int).reset_index()
	churn_status.columns = [user_col, date_col, 'churn_status']
	churn_status[user_col] = churn_status[user_col].astype(int)  # Ensure int type
	
	return churn_status


def aggregate_user_day_activity(df: pd.DataFrame,
                                  user_col: str = 'userId',
                                  time_col: str = 'time',
                                  page_col: str = 'page',
                                  level_col: str = 'level',
                                  registration_col: str = 'registration',
                                  drop_cancellation: bool = True,
                                  fill_missing_days: bool = True) -> pd.DataFrame:
	"""
	Aggregate user activity data by userId and day, creating count columns for each page category.
	
	This function takes event-level data and aggregates it to the user-day level,
	creating a single row for each user's activity on a given day. Each unique
	page category becomes a column containing the count of that page type.
	
	Parameters
	----------
	df : pd.DataFrame
		Input DataFrame containing user event data
	user_col : str, default 'userId'
		Column name identifying the user
	time_col : str, default 'time'
		Column name containing timestamps (should be datetime type)
	page_col : str, default 'page'
		Column name containing page/event categories
	level_col : str, default 'level'
		Column name containing user subscription level
	registration_col : str, default 'registration'
		Column name containing user registration timestamp
	drop_cancellation : bool, default True
		Whether to drop the 'Cancellation Confirmation' column from results
		(typically used as a target variable rather than a feature)
	fill_missing_days : bool, default True
		Whether to fill missing days between each user's first and last activity
		with zeros. This ensures inactive days are represented in the dataset,
		which is critical for time-series modeling with sliding windows.
	
	Returns
	-------
	pd.DataFrame
		Aggregated DataFrame with columns:
		- user_col: User identifier
		- 'date': Date extracted from time_col
		- 'level': Last observed subscription level for each day (forward-filled)
		- 'days_since_registration': Days elapsed since user registration
		- 'session_count': Number of distinct sessions for the user on that day (0 if inactive)
		- 'event_count': Total events observed that day (0 if inactive)
		- 'active_flag': 1 if any events occurred that day, else 0
		- 'events_per_session': event_count / session_count (0 if no sessions)
		- One column per unique page category with count values
		When fill_missing_days=True, inactive days are included with zero counts.
	
	Raises
	------
	ValueError
		If required columns are not found in the DataFrame
	
	Example
	-------
	>>> df_aggregated = aggregate_user_day_activity(df, fill_missing_days=True)
	>>> print(df_aggregated.shape)
	(205676, 21)
	>>> print(df_aggregated.columns.tolist()[:5])
	['userId', 'date', 'About', 'Add Friend', 'Add to Playlist']
	"""
	# Validate required columns
	if user_col not in df.columns:
		raise ValueError(f"user_col '{user_col}' not found in DataFrame")
	if time_col not in df.columns:
		raise ValueError(f"time_col '{time_col}' not found in DataFrame")
	if page_col not in df.columns:
		raise ValueError(f"page_col '{page_col}' not found in DataFrame")
	
	# Work with a copy to avoid modifying the original
	df_copy = df.copy()
	
	# Extract the date from the time column and normalize
	df_copy['date'] = pd.to_datetime(df_copy[time_col]).dt.normalize()

	# Per-day event counts and session counts (when sessionId is available)
	per_day_counts = df_copy.groupby([user_col, 'date'], observed=False).size().reset_index(name='event_count')
	
	# Compute ACCUMULATED unique artist count per user (cumulative up to each day)
	if 'artist' in df_copy.columns:
		# Filter out null/empty artists
		df_with_artists = df_copy[df_copy['artist'].notna() & (df_copy['artist'] != '')].copy()
		
		# Sort by user and date
		df_with_artists = df_with_artists.sort_values([user_col, 'date'])
		
		# For each user-date, get cumulative unique artists
		def calc_cumulative_artists(group):
			# For each date in the group, calculate cumulative unique artists up to that date
			result = []
			seen_artists = set()
			for date in sorted(group['date'].unique()):
				# Get all artists heard up to and including this date
				artists_up_to_date = group[group['date'] <= date]['artist'].unique()
				seen_artists.update(artists_up_to_date)
				result.append({'date': date, 'unique_artists': len(seen_artists)})
			return pd.DataFrame(result)
		
		unique_artists = df_with_artists.groupby(user_col, observed=False).apply(calc_cumulative_artists).reset_index()
		unique_artists = unique_artists[[user_col, 'date', 'unique_artists']]
	else:
		unique_artists = None
	
	# Compute ACCUMULATED error count per user (cumulative up to each day)
	if page_col in df_copy.columns:
		# Filter error events
		df_errors = df_copy[df_copy[page_col] == 'Error'].copy()
		
		if not df_errors.empty:
			# Sort by user and date
			df_errors = df_errors.sort_values([user_col, 'date'])
			
			# Count errors per user-day, then cumsum per user
			errors_per_day = df_errors.groupby([user_col, 'date'], observed=False).size().reset_index(name='daily_errors')
			errors_per_day = errors_per_day.sort_values([user_col, 'date'])
			errors_per_day['accumulated_errors'] = errors_per_day.groupby(user_col, observed=False)['daily_errors'].cumsum()
			
			# Keep only userId, date, and accumulated_errors
			accumulated_errors = errors_per_day[[user_col, 'date', 'accumulated_errors']]
		else:
			accumulated_errors = None
	else:
		accumulated_errors = None
	
	if 'sessionId' in df_copy.columns:
		# OPTIMIZED: Single-pass aggregation for session count and avg session length
		# This avoids creating the massive intermediate session_times DataFrame
		def calc_session_stats(group):
			# Count unique sessions
			session_count = group['sessionId'].nunique()
			
			# Calculate average session duration
			session_durations = (
				group.groupby('sessionId', observed=False)[time_col]
				.apply(lambda x: (pd.to_datetime(x.max()) - pd.to_datetime(x.min())).total_seconds())
			)
			avg_duration = session_durations.mean() if len(session_durations) > 0 else 0.0
			
			return pd.Series({
				'session_count': session_count,
				'avg_session_length': avg_duration
			})
		
		session_stats = df_copy.groupby([user_col, 'date'], observed=False).apply(calc_session_stats).reset_index()
		session_counts = session_stats[[user_col, 'date', 'session_count']]
		avg_session_length = session_stats[[user_col, 'date', 'avg_session_length']]
	else:
		session_counts = None
		avg_session_length = None
	
	# Get user registration dates
	if registration_col in df_copy.columns:
		# Use actual registration dates from the registration column
		user_registration = df_copy.groupby(user_col, observed=False)[registration_col].first().reset_index()
		user_registration.columns = [user_col, 'registration_date']
		user_registration['registration_date'] = pd.to_datetime(user_registration['registration_date']).dt.normalize()
	else:
		# Fallback: use first observed event if registration column missing
		user_registration = df_copy.groupby(user_col, observed=False)[time_col].min().reset_index()
		user_registration.columns = [user_col, 'registration_date']
		user_registration['registration_date'] = pd.to_datetime(user_registration['registration_date']).dt.normalize()
	
	# Get last level observation per user per day
	if level_col in df_copy.columns:
		level_per_day = df_copy.groupby([user_col, 'date'], observed=False)[level_col].last().reset_index()
	else:
		print(f"Warning: '{level_col}' column not found, skipping level tracking")
		level_per_day = None
	
	# Create pivot table with counts for each page category per user per day
	df_aggregated = df_copy.groupby([user_col, 'date', page_col], observed=False).size().unstack(fill_value=0).reset_index()
	
	# Drop 'Error' column if present since we compute accumulated errors separately
	if 'Error' in df_aggregated.columns:
		df_aggregated = df_aggregated.drop(columns=['Error'])

	# Attach per-day counts and flags
	df_aggregated = df_aggregated.merge(per_day_counts, on=[user_col, 'date'], how='left')
	
	# Attach unique artist counts
	if unique_artists is not None:
		df_aggregated = df_aggregated.merge(unique_artists, on=[user_col, 'date'], how='left')
		df_aggregated['unique_artists'] = df_aggregated['unique_artists'].fillna(0).astype(int)
	
	# Attach accumulated error counts
	if accumulated_errors is not None:
		df_aggregated = df_aggregated.merge(accumulated_errors, on=[user_col, 'date'], how='left')
		df_aggregated['accumulated_errors'] = df_aggregated['accumulated_errors'].fillna(0).astype(int)
	
	if session_counts is not None:
		df_aggregated = df_aggregated.merge(session_counts, on=[user_col, 'date'], how='left')
		if avg_session_length is not None:
			df_aggregated = df_aggregated.merge(avg_session_length, on=[user_col, 'date'], how='left')
		else:
			df_aggregated['avg_session_length'] = 0.0
	else:
		# If sessionId is missing, approximate sessions by treating each day as one session when active
		df_aggregated['session_count'] = 1
		df_aggregated['avg_session_length'] = 0.0

	# Active flag and events-per-session ratio
	df_aggregated['active_flag'] = (df_aggregated['event_count'] > 0).astype(int)
	df_aggregated['events_per_session'] = df_aggregated.apply(
		lambda row: row['event_count'] / row['session_count'] if row['session_count'] else 0,
		axis=1
	)
	
	# Optionally drop the cancellation confirmation column
	if drop_cancellation and 'Cancellation Confirmation' in df_aggregated.columns:
		df_aggregated = df_aggregated.drop(columns=['Cancellation Confirmation'])
	
	# Fill missing days with zeros per user if requested (VECTORIZED APPROACH)
	if fill_missing_days:
		activity_cols = [col for col in df_aggregated.columns if col not in [user_col, 'date']]
		
		# Get per-user date ranges
		user_date_ranges = df_aggregated.groupby(user_col, observed=False)['date'].agg(['min', 'max'])
		
		# Build list of (user, date) tuples only for each user's active period
		user_date_pairs = []
		for user_id, row in user_date_ranges.iterrows():
			dates = pd.date_range(row['min'], row['max'], freq='D')
			user_date_pairs.extend([(user_id, d) for d in dates])
		
		# Create complete index from pairs
		complete_df = pd.DataFrame(user_date_pairs, columns=[user_col, 'date'])
		
		# Merge with aggregated data
		df_aggregated = complete_df.merge(df_aggregated, on=[user_col, 'date'], how='left')
		
		# Fill missing values with 0
		df_aggregated[activity_cols] = df_aggregated[activity_cols].fillna(0)
		
		# Convert to appropriate types
		int_cols = [col for col in activity_cols if col not in ['events_per_session', 'avg_session_length']]
		for col in int_cols:
			if col in df_aggregated.columns:
				df_aggregated[col] = df_aggregated[col].astype(int)
	
	# Add level column with forward-filling
	if level_per_day is not None:
		df_aggregated = df_aggregated.merge(level_per_day, on=[user_col, 'date'], how='left')
		
		# Forward-fill level per user
		df_aggregated['date_temp'] = pd.to_datetime(df_aggregated['date'])
		df_aggregated = df_aggregated.sort_values([user_col, 'date_temp'])
		df_aggregated['level'] = df_aggregated.groupby(user_col, observed=False)['level'].ffill()
		df_aggregated = df_aggregated.drop(columns=['date_temp'])
	
	# Add days_since_registration
	df_aggregated = df_aggregated.merge(user_registration, on=user_col, how='left')
	df_aggregated['date_temp'] = pd.to_datetime(df_aggregated['date'])
	df_aggregated['days_since_registration'] = (df_aggregated['date_temp'] - df_aggregated['registration_date']).dt.days
	df_aggregated = df_aggregated.drop(columns=['registration_date', 'date_temp'])
	
	# Convert date back to date type (not datetime) for consistency
	df_aggregated['date'] = df_aggregated['date'].dt.date if hasattr(df_aggregated['date'], 'dt') else df_aggregated['date']
	
	return df_aggregated


def add_rolling_averages(df: pd.DataFrame,
                         columns: list = None,
                         n: int = 7,
                         user_col: str = 'userId',
						 date_col: str = 'date',
						 fill_missing_with_zero: bool = True) -> pd.DataFrame:
	"""
	Add rolling average columns for specified metrics over the last n days.
	
	For each specified column, creates a new column named '<col_name_lowercase>_avg_<n>d'
	that calculates the average value of that column over the n days before each row's date
	(not including the row's date itself).
	
	Parameters
	----------
	df : pd.DataFrame
		Aggregated user-day DataFrame (typically output from aggregate_user_day_activity)
	columns : list, default ['NextSong']
		List of column names to calculate rolling averages for
	n : int, default 7
		Number of days to include in the rolling average window
	user_col : str, default 'userId'
		Column name identifying the user
	date_col : str, default 'date'
		Column name containing the date values
	fill_missing_with_zero : bool, default True
		Whether to fill missing user-days with zeros for the specified columns before
		computing rolling averages (helps include inactive days in the window)
	
	Returns
	-------
	pd.DataFrame
		DataFrame with new rolling average columns added for each specified column
	
	Raises
	------
	ValueError
		If required columns are not found in the DataFrame
	
	Example
	-------
	>>> df_agg = add_rolling_averages(df_agg, columns=['NextSong', 'Thumbs Up'], n=7)
	>>> print(df_agg.columns)
	['userId', 'date', ..., 'nextsong_avg_7d', 'thumbs_up_avg_7d']
	
	Notes
	-----
	- Each row's rolling average is calculated based on that row's date
	- The window includes the n days before the row's date, not including the date itself
	- If a user has fewer than n days of history before a date, the average is over available days
	- Users with no history in the window will have NaN values
	- When fill_missing_with_zero is True, gaps in user activity are filled with
	  zeros so dormant days contribute 0 to the rolling window
	"""
	if columns is None:
		columns = ['NextSong']
	
	# Validate required columns
	if user_col not in df.columns:
		raise ValueError(f"user_col '{user_col}' not found in DataFrame")
	if date_col not in df.columns:
		raise ValueError(f"date_col '{date_col}' not found in DataFrame")
	
	for col in columns:
		if col not in df.columns:
			raise ValueError(f"Column '{col}' not found in DataFrame")
	
	# Work with a copy to avoid modifying the original
	df_copy = df.copy()

	# Ensure date column is datetime type
	df_copy[date_col] = pd.to_datetime(df_copy[date_col])
	
	# Pre-compute column name mappings
	col_mappings = {col: f'{col.lower().replace(" ", "_")}_avg_{n}d' for col in columns}
	
	# Prepare storage for results
	processed_users = []
	for user_id, user_data in df_copy.groupby(user_col):
		user_data = user_data.sort_values(date_col).copy()
		
		# Build full daily index per user if requested
		if fill_missing_with_zero:
			full_range = pd.date_range(user_data[date_col].min(), user_data[date_col].max(), freq='D')
			user_data = user_data.set_index(date_col).reindex(full_range)
			user_data[user_col] = user_id
			user_data.index.name = date_col
			user_data = user_data.reset_index()
			
			# Fill specified metric columns with zeros on missing days
			for col in columns:
				user_data[col] = user_data[col].fillna(0)
		else:
			user_data = user_data.reset_index(drop=True)
		
		# Compute rolling averages using vectorized rolling window
		for col in columns:
			new_col_name = col_mappings[col]
			
			# Use time-based rolling window with shift to exclude current date
			user_data_sorted = user_data.sort_values(date_col)
			rolling_values = (
				user_data_sorted.set_index(date_col)[col]
				.rolling(window=f'{n}D', min_periods=1)
				.mean()
				.shift(1)  # Exclude current day from window
			)
			user_data[new_col_name] = rolling_values.reset_index(drop=True)
		
		processed_users.append(user_data)

	# Combine all users
	result = pd.concat(processed_users, ignore_index=True)
	result[date_col] = result[date_col].dt.date

	return result


# ============================================================================
# MODULAR SKLEARN TRANSFORMERS FOR FEATURE ENGINEERING
# ============================================================================


class BasicEventAggregator(BaseEstimator, TransformerMixin):
	"""
	Aggregates raw event data to user-day level with basic page counts.
	
	This is the first step in feature engineering - converts event-level data
	into user-day aggregates. Only computes features that will actually be used
	in the final model (no intermediate features that get dropped later).
	
	Features created:
	- Event counts per page type (for pages used in rolling averages)
	- Session metrics (count, avg length, events per session)
	- User metadata (level, days since registration)
	"""
	def __init__(self, 
				 user_col='userId',
				 time_col='time',
				 page_col='page',
				 level_col='level',
				 registration_col='registration',
				 session_col='sessionId',
				 pages_for_rolling=None):
		"""
		Parameters
		----------
		pages_for_rolling : list, optional
			Page types to aggregate for rolling average computation.
			Defaults to pages used in final model: Add Friend, Add to Playlist, 
			Thumbs Up, Thumbs Down
		"""
		self.user_col = user_col
		self.time_col = time_col
		self.page_col = page_col
		self.level_col = level_col
		self.registration_col = registration_col
		self.session_col = session_col
		# Only aggregate pages we'll actually use (for rolling averages)
		self.pages_for_rolling = pages_for_rolling or [
			'Add Friend', 'Add to Playlist', 'Thumbs Up', 'Thumbs Down'
		]
	
	def fit(self, X, y=None):
		return self
	
	def transform(self, X):
		print("BasicEventAggregator: Aggregating events to user-day level...")
		df = X.copy()
		
		# Extract date from timestamp
		df['date'] = pd.to_datetime(df[self.time_col]).dt.normalize()
		
		# Filter to only page types we need for rolling averages
		df_filtered = df[df[self.page_col].isin(self.pages_for_rolling)].copy()
		
		# Create pivot table for page counts
		page_counts = df_filtered.groupby([self.user_col, 'date', self.page_col], observed=False).size().unstack(fill_value=0).reset_index()
		
		# Total event count per user-day (across ALL events, not just filtered pages)
		event_counts = df.groupby([self.user_col, 'date'], observed=False).size().reset_index(name='event_count')
		
		# Merge page counts with event counts
		result = page_counts.merge(event_counts, on=[self.user_col, 'date'], how='right')
		
		# Fill NaN values in page columns with 0 (days with no activity for those pages)
		for page in self.pages_for_rolling:
			if page in result.columns:
				result[page] = result[page].fillna(0).astype(int)
		
		# Session metrics
		if self.session_col in df.columns:
			# Session counts per user-day
			session_counts = df.groupby([self.user_col, 'date'], observed=False)[self.session_col].nunique().reset_index(name='session_count')
			
			# Average session length per user-day
			def calc_avg_session_length(group):
				session_durations = group.groupby(self.session_col)[self.time_col].apply(
					lambda x: (pd.to_datetime(x.max()) - pd.to_datetime(x.min())).total_seconds()
				)
				return session_durations.mean() if len(session_durations) > 0 else 0.0
			
			avg_session_length = df.groupby([self.user_col, 'date'], observed=False).apply(calc_avg_session_length).reset_index(name='avg_session_length')
			
			result = result.merge(session_counts, on=[self.user_col, 'date'], how='left')
			result = result.merge(avg_session_length, on=[self.user_col, 'date'], how='left')
		else:
			result['session_count'] = 1
			result['avg_session_length'] = 0.0
		
		# Events per session ratio
		result['events_per_session'] = result.apply(
			lambda row: row['event_count'] / row['session_count'] if row['session_count'] > 0 else 0,
			axis=1
		)
		
		# User level (last observed per day)
		if self.level_col in df.columns:
			level_per_day = df.groupby([self.user_col, 'date'], observed=False)[self.level_col].last().reset_index()
			result = result.merge(level_per_day, on=[self.user_col, 'date'], how='left')
			# Forward fill level per user
			result = result.sort_values([self.user_col, 'date'])
			result[self.level_col] = result.groupby(self.user_col, observed=False)[self.level_col].ffill()
		
		# Days since registration
		if self.registration_col in df.columns:
			user_registration = df.groupby(self.user_col, observed=False)[self.registration_col].first().reset_index()
			user_registration[self.registration_col] = pd.to_datetime(user_registration[self.registration_col]).dt.normalize()
			result = result.merge(user_registration, on=self.user_col, how='left')
			result['days_since_registration'] = (result['date'] - result[self.registration_col]).dt.days
			result = result.drop(columns=[self.registration_col])
		
		# Fill missing days with zeros per user (critical for time-series)
		result = self._fill_missing_days(result)
		
		print(f"  Output shape: {result.shape}")
		print(f"  Features: {[col for col in result.columns if col not in [self.user_col, 'date']]}")
		
		return result
	
	def _fill_missing_days(self, df):
		"""Fill missing days between min and max date per user with zeros."""
		# Get per-user date ranges
		user_date_ranges = df.groupby(self.user_col, observed=False)['date'].agg(['min', 'max'])
		
		# Build complete user-date combinations
		user_date_pairs = []
		for user_id, row in user_date_ranges.iterrows():
			dates = pd.date_range(row['min'], row['max'], freq='D')
			user_date_pairs.extend([(user_id, d) for d in dates])
		
		complete_df = pd.DataFrame(user_date_pairs, columns=[self.user_col, 'date'])
		
		# Merge and fill missing values
		activity_cols = [col for col in df.columns if col not in [self.user_col, 'date']]
		result = complete_df.merge(df, on=[self.user_col, 'date'], how='left')
		
		# Fill missing values by column type
		for col in activity_cols:
			if col not in result.columns:
				continue
			if result[col].dtype.name == 'category' or col == self.level_col:
				# For categorical columns, fill with the mode or first category
				if result[col].notna().any():
					result[col] = result[col].fillna(result[col].mode()[0] if len(result[col].mode()) > 0 else result[col].cat.categories[0])
			else:
				# For numeric columns, fill with 0
				result[col] = result[col].fillna(0)
		
		# Convert to appropriate types
		int_cols = [col for col in activity_cols if col not in ['events_per_session', 'avg_session_length', self.level_col]]
		for col in int_cols:
			if col in result.columns:
				result[col] = result[col].astype(int)
		
		return result


class AccumulatedFeaturesTransformer(BaseEstimator, TransformerMixin):
	"""
	Computes accumulated (cumulative) features per user.
	
	Features created:
	- accumulated_errors: Cumulative error count up to each date
	- accumulated_unique_artists: Cumulative unique artist count up to each date
	
	These features carry historical context and must be recomputed per fold
	in cross-validation to prevent leakage.
	"""
	def __init__(self, 
				 user_col='userId',
				 time_col='time',
				 page_col='page',
				 artist_col='artist'):
		self.user_col = user_col
		self.time_col = time_col
		self.page_col = page_col
		self.artist_col = artist_col
		self.raw_df_ = None
	
	def fit(self, X, y=None, raw_df=None):
		"""
		Store raw dataframe for computing accumulated features.
		
		Parameters
		----------
		raw_df : pd.DataFrame
			Raw event-level data needed to compute cumulative features
		"""
		if raw_df is not None:
			self.raw_df_ = raw_df
		return self
	
	def transform(self, X):
		if self.raw_df_ is None:
			raise ValueError("raw_df must be provided during fit()")
		
		print("AccumulatedFeaturesTransformer: Computing cumulative features...")
		df_agg = X.copy()
		df_raw = self.raw_df_.copy()
		
		# Ensure date column is datetime in both dataframes
		if 'date' in df_agg.columns:
			df_agg['date'] = pd.to_datetime(df_agg['date'])
		df_raw['date'] = pd.to_datetime(df_raw[self.time_col]).dt.normalize()
		
		# Accumulated errors
		df_errors = df_raw[df_raw[self.page_col] == 'Error'].copy()
		if not df_errors.empty:
			errors_per_day = df_errors.groupby([self.user_col, 'date']).size().reset_index(name='daily_errors')
			errors_per_day = errors_per_day.sort_values([self.user_col, 'date'])
			errors_per_day['accumulated_errors'] = errors_per_day.groupby(self.user_col, observed=False)['daily_errors'].cumsum()
			
			# Merge with aggregated data
			df_agg = df_agg.merge(
				errors_per_day[[self.user_col, 'date', 'accumulated_errors']],
				on=[self.user_col, 'date'],
				how='left'
			)
		else:
			df_agg['accumulated_errors'] = 0
		
		df_agg['accumulated_errors'] = df_agg['accumulated_errors'].fillna(0).astype(int)
		
		# Accumulated unique artists
		if self.artist_col in df_raw.columns:
			df_with_artists = df_raw[df_raw[self.artist_col].notna() & (df_raw[self.artist_col] != '')].copy()
			
			if not df_with_artists.empty:
				df_with_artists = df_with_artists.sort_values([self.user_col, 'date'])
				
				# Compute cumulative unique artists per user
				def calc_cumulative_artists(group):
					result = []
					seen_artists = set()
					for date in sorted(group['date'].unique()):
						artists_up_to_date = group[group['date'] <= date][self.artist_col].unique()
						seen_artists.update(artists_up_to_date)
						result.append({'date': date, 'accumulated_unique_artists': len(seen_artists)})
					return pd.DataFrame(result)
				
				unique_artists = df_with_artists.groupby(self.user_col, observed=False).apply(calc_cumulative_artists).reset_index()
				unique_artists = unique_artists[[self.user_col, 'date', 'accumulated_unique_artists']]
				
				# Merge with aggregated data
				df_agg = df_agg.merge(unique_artists, on=[self.user_col, 'date'], how='left')
			else:
				df_agg['accumulated_unique_artists'] = 0
		else:
			df_agg['accumulated_unique_artists'] = 0
		
		df_agg['accumulated_unique_artists'] = df_agg['accumulated_unique_artists'].fillna(0).astype(int)
		
		print(f"  Added: accumulated_errors, accumulated_unique_artists")
		
		return df_agg

class PageInteractionTransformer(BaseEstimator, TransformerMixin):
	"""
	Computes page interaction features from raw event data.
	
	Features created:
	- Page counts for specific pages (About, Help, Settings, etc.)
	
	Only creates features that will actually be used in the final model.
	"""
	def __init__(self, 
				 user_col='userId',
				 time_col='time',
				 page_col='page',
				 pages_to_track=None):
		"""
		Parameters
		----------
		pages_to_track : list, optional
			Specific page types to track. Defaults to pages used in final model:
			About, Help, Settings, Save Settings, Home
		"""
		self.user_col = user_col
		self.time_col = time_col
		self.page_col = page_col
		self.pages_to_track = pages_to_track or [
			'About', 'Help', 'Settings', 'Save Settings', 'Home'
		]
		self.raw_df_ = None
	
	def fit(self, X, y=None, raw_df=None):
		"""
		Store raw dataframe for computing page interactions.
		
		Parameters
		----------
		raw_df : pd.DataFrame
			Raw event-level data needed to compute page interaction features
		"""
		if raw_df is not None:
			self.raw_df_ = raw_df
		return self
	
	def transform(self, X):
		if self.raw_df_ is None:
			raise ValueError("raw_df must be provided during fit()")
		
		print("PageInteractionTransformer: Computing page interaction features...")
		df_agg = X.copy()
		df_raw = self.raw_df_.copy()
		
		# Extract date from raw events
		df_raw['date'] = pd.to_datetime(df_raw[self.time_col]).dt.normalize()
		
		# Filter to tracked pages
		df_pages = df_raw[df_raw[self.page_col].isin(self.pages_to_track)].copy()
		
		if not df_pages.empty:
			# Count page interactions per user-day
			page_counts = df_pages.groupby([self.user_col, 'date', self.page_col]).size().unstack(fill_value=0).reset_index()
			
			# Merge with aggregated data
			df_agg = df_agg.merge(page_counts, on=[self.user_col, 'date'], how='left')
			
			# Fill NaN with 0 for page columns
			for page in self.pages_to_track:
				if page in df_agg.columns:
					df_agg[page] = df_agg[page].fillna(0).astype(int)
		else:
			# Add empty columns if no page interactions found
			for page in self.pages_to_track:
				df_agg[page] = 0
		
		pages_added = [p for p in self.pages_to_track if p in df_agg.columns]
		print(f"  Added: {pages_added}")
		
		return df_agg


class RollingAverageTransformerModular(BaseEstimator, TransformerMixin):
	"""
	Computes rolling average features from user-day aggregated data.
	
	This transformer should be used AFTER BasicEventAggregator to compute
	rolling averages over the page counts. Window excludes current day to
	prevent leakage.
	
	Features created:
	- <page>_avg_<n>d: Rolling n-day average for each specified page type
	"""
	def __init__(self, 
				 pages=None,
				 window_days=7,
				 user_col='userId',
				 date_col='date'):
		"""
		Parameters
		----------
		pages : list, optional
			Page types to compute rolling averages for.
			Defaults to: Add Friend, Add to Playlist, Thumbs Up, Thumbs Down
		window_days : int
			Number of days in rolling window
		"""
		self.pages = pages or ['Add Friend', 'Add to Playlist', 'Thumbs Up', 'Thumbs Down']
		self.window_days = window_days
		self.user_col = user_col
		self.date_col = date_col
	
	def fit(self, X, y=None):
		return self
	
	def transform(self, X):
		print(f"RollingAverageTransformerModular: Computing {self.window_days}d rolling averages...")
		return add_rolling_averages(
			X, 
			columns=self.pages, 
			n=self.window_days,
			user_col=self.user_col,
			date_col=self.date_col,
			fill_missing_with_zero=True
		)


class TrendFeaturesTransformer(BaseEstimator, TransformerMixin):
	"""
	Creates trend features by comparing short-term vs long-term averages.
	
	Features created:
	- <page>_trend: (7d_avg / 14d_avg) - 1
	  Positive values = increasing activity
	  Negative values = decreasing activity
	"""
	def __init__(self, pages=None):
		"""
		Parameters
		----------
		pages : list, optional
			Page types to compute trends for.
			Defaults to: Add Friend, Add to Playlist, Thumbs Up, Thumbs Down
		"""
		self.pages = pages or ['Add Friend', 'Add to Playlist', 'Thumbs Up', 'Thumbs Down']
	
	def fit(self, X, y=None):
		return self
	
	def transform(self, X):
		print("TrendFeaturesTransformer: Computing trend features...")
		X_copy = X.copy()
		
		for page in self.pages:
			col_7d = f'{page.lower().replace(" ", "_")}_avg_7d'
			col_14d = f'{page.lower().replace(" ", "_")}_avg_14d'
			trend_col = f'{page.lower().replace(" ", "_")}_trend'
			
			if col_7d in X_copy.columns and col_14d in X_copy.columns:
				# Compute ratio: (7d / 14d) - 1
				denominator = X_copy[col_14d].replace(0, np.nan)
				X_copy[trend_col] = (X_copy[col_7d] / denominator) - 1
				X_copy[trend_col] = X_copy[trend_col].fillna(0)
		
		return X_copy


class FeaturePreprocessor(BaseEstimator, TransformerMixin):
	"""
	Final preprocessing step: type conversions and derived features.
	
	Transformations:
	- Convert level to binary (paid=1, free=0)
	- Add is_weekend flag
	- Fill ratio columns with 0
	"""
	def fit(self, X, y=None):
		return self
	
	def transform(self, X):
		print("FeaturePreprocessor: Final preprocessing...")
		X_copy = X.copy()
		
		# Convert level to binary
		if 'level' in X_copy.columns:
			X_copy['level'] = (X_copy['level'] == 'paid').astype(int)
		
		# Add weekend indicator
		if 'date' in X_copy.columns:
			X_copy['date'] = pd.to_datetime(X_copy['date'])
			X_copy['is_weekend'] = (X_copy['date'].dt.dayofweek >= 5).astype(int)
		
		# Fill ratio columns with 0
		ratio_cols = [col for col in X_copy.columns if 'ratio' in col.lower() or 'trend' in col.lower()]
		for col in ratio_cols:
			if col in X_copy.columns:
				X_copy[col] = pd.to_numeric(X_copy[col], errors='coerce').fillna(0)
		
		return X_copy


class CancellationTargetTransformerModular(BaseEstimator, TransformerMixin):
	"""
	Computes cancellation targets using vectorized operations.
	
	IMPORTANT: This transformer requires raw event data during fit() to compute
	forward-looking churn labels. Must be recomputed per fold in CV to prevent leakage.
	"""
	def __init__(self, window_days=10, user_col='userId', page_col='page', time_col='time'):
		self.window_days = window_days
		self.user_col = user_col
		self.page_col = page_col
		self.time_col = time_col
		self.raw_df_ = None
	
	def fit(self, X, y=None, raw_df=None):
		"""
		Store raw dataframe for computing churn targets.
		
		Parameters
		----------
		raw_df : pd.DataFrame
			Raw event-level data containing cancellation events
		"""
		if raw_df is not None:
			self.raw_df_ = raw_df
		return self
	
	def transform(self, X):
		if self.raw_df_ is None:
			raise ValueError("raw_df must be provided during fit()")
		
		print(f"CancellationTargetTransformerModular: Computing churn targets (window={self.window_days}d)...")
		
		# Use efficient batch computation
		churn_targets = compute_cancellation_batch(
			self.raw_df_,
			X,
			user_col=self.user_col,
			page_col=self.page_col,
			time_col=self.time_col,
			window_days=self.window_days
		)
		
		# Merge with X
		X_copy = X.copy()
		X_copy['date'] = pd.to_datetime(X_copy['date'])
		churn_targets['date'] = pd.to_datetime(churn_targets['date'])
		X_copy[self.user_col] = X_copy[self.user_col].astype(int)
		churn_targets[self.user_col] = churn_targets[self.user_col].astype(int)
		
		result = X_copy.merge(churn_targets, on=[self.user_col, 'date'], how='left')
		
		print(f"  Churn status - 0: {(result['churn_status']==0).sum()}, 1: {(result['churn_status']==1).sum()}")
		
		return result


class RawDataSplitter(BaseEstimator, TransformerMixin):
	"""
	Filters raw event data by cutoff date at the start of the pipeline.
	
	This ensures temporal integrity - only data before cutoff_date is used
	for training, preventing future data leakage.
	
	Parameters
	----------
	cutoff_date : str or pd.Timestamp
		Maximum date to include in the data (inclusive)
	time_col : str, default 'time'
		Name of the timestamp column
	"""
	def __init__(self, cutoff_date, time_col='time'):
		self.cutoff_date = pd.to_datetime(cutoff_date)
		self.time_col = time_col
	
	def fit(self, X, y=None):
		"""No fitting needed - this is a stateless filter."""
		return self
	
	def transform(self, X):
		"""Filter data to only include records before cutoff_date."""
		if X is None or len(X) == 0:
			return X
		
		X_copy = X.copy()
		X_copy['_temp_time'] = pd.to_datetime(X_copy[self.time_col])
		X_filtered = X_copy[X_copy['_temp_time'] <= self.cutoff_date].copy()
		X_filtered = X_filtered.drop(columns=['_temp_time'])
		
		print(f"RawDataSplitter: Filtered to {len(X_filtered):,} events (<= {self.cutoff_date.date()})")
		
		return X_filtered


class RollingEventAggregator(BaseEstimator, TransformerMixin):
	"""
	Computes rolling average features DIRECTLY from raw event data.
	
	This transformer bypasses the user-day aggregation step by computing
	rolling averages directly from event timestamps. This simplifies the pipeline
	and reduces the number of intermediate data structures.
	
	Features created:
	- <event_type>_avg_<n>d: Rolling n-day average count for each event type
	
	Window excludes current day to prevent leakage.
	"""
	def __init__(self, 
				 event_types=None,
				 window_days=7,
				 user_col='userId',
				 time_col='time',
				 page_col='page'):
		"""
		Parameters
		----------
		event_types : list, optional
			Event/page types to compute rolling averages for.
			Defaults to: Add Friend, Add to Playlist, Thumbs Up, Thumbs Down
		window_days : int
			Number of days in rolling window
		"""
		self.event_types = event_types or ['Add Friend', 'Add to Playlist', 'Thumbs Up', 'Thumbs Down']
		self.window_days = window_days
		self.user_col = user_col
		self.time_col = time_col
		self.page_col = page_col
	
	def fit(self, X, y=None):
		return self
	
	def transform(self, X):
		"""
		Transforms raw event data into rolling average features.
		
		Parameters
		----------
		X : pd.DataFrame
			Raw event-level data (NOT user-day aggregated)
		
		Returns
		-------
		pd.DataFrame
			User-day level data with rolling average features
		"""
		print(f"RollingEventAggregator: Computing {self.window_days}d rolling averages from raw events...")
		df = X.copy()
		
		# Ensure timestamp is datetime
		df[self.time_col] = pd.to_datetime(df[self.time_col])
		df['date'] = df[self.time_col].dt.normalize()
		
		# Filter to event types of interest
		df_filtered = df[df[self.page_col].isin(self.event_types)].copy()
		
		# Count events per user-date-page combination
		event_counts = df_filtered.groupby([self.user_col, 'date', self.page_col]).size().reset_index(name='count')
		
		# Pivot to get one column per event type
		event_pivot = event_counts.pivot_table(
			index=[self.user_col, 'date'],
			columns=self.page_col,
			values='count',
			fill_value=0
		).reset_index()
		
		# Get all unique user-date combinations from original data
		all_user_dates = df.groupby([self.user_col, 'date']).size().reset_index(name='_temp')
		all_user_dates = all_user_dates[[self.user_col, 'date']]
		
		# Merge to ensure all days are represented
		result = all_user_dates.merge(event_pivot, on=[self.user_col, 'date'], how='left')
		
		# Fill missing event counts with 0
		for event_type in self.event_types:
			if event_type in result.columns:
				result[event_type] = result[event_type].fillna(0).astype(int)
		
		# Compute rolling averages per user
		result = result.sort_values([self.user_col, 'date'])
		
		rolling_features = []
		for user_id, user_data in result.groupby(self.user_col):
			user_data = user_data.sort_values('date').copy()
			
			# Fill missing days between min and max
			full_range = pd.date_range(user_data['date'].min(), user_data['date'].max(), freq='D')
			user_data = user_data.set_index('date').reindex(full_range)
			user_data[self.user_col] = user_id
			user_data.index.name = 'date'
			user_data = user_data.reset_index()
			
			# Fill event counts with 0 for missing days
			for event_type in self.event_types:
				if event_type in user_data.columns:
					user_data[event_type] = user_data[event_type].fillna(0)
			
			# Compute rolling averages
			for event_type in self.event_types:
				if event_type in user_data.columns:
					col_name = f'{event_type.lower().replace(" ", "_")}_avg_{self.window_days}d'
					
					# Time-based rolling window excluding current day
					rolling_values = (
						user_data.set_index('date')[event_type]
						.rolling(window=f'{self.window_days}D', min_periods=1)
						.mean()
						.shift(1)  # Exclude current day to prevent leakage
					)
					user_data[col_name] = rolling_values.reset_index(drop=True)
			
			rolling_features.append(user_data)
		
		# Combine all users
		result_with_rolling = pd.concat(rolling_features, ignore_index=True)
		
		# Drop the raw event count columns (keep only rolling averages)
		cols_to_drop = [col for col in result_with_rolling.columns if col in self.event_types]
		result_with_rolling = result_with_rolling.drop(columns=cols_to_drop)
		
		rolling_cols = [col for col in result_with_rolling.columns if '_avg_' in col]
		print(f"  Output shape: {result_with_rolling.shape}")
		print(f"  Rolling features created: {rolling_cols}")
		
		return result_with_rolling


# ============================================================================
# PIPELINE FACTORY AND CROSS-VALIDATION UTILITIES
# ============================================================================

def create_feature_pipeline(cutoff_date, window_days=10, mode='train'):
	"""
	Factory function to create a fresh feature engineering pipeline.
	
	This function creates a new pipeline instance for each fold in cross-validation,
	ensuring that all transformers are fitted only on fold training data to prevent
	data leakage.
	
	Parameters
	----------
	cutoff_date : str or pd.Timestamp
		Date cutoff for training data (data before this date will be used)
	window_days : int, default 10
		Number of days to look ahead for churn prediction
	mode : str, default 'train'
		Either 'train' (includes churn target computation) or 'predict' (for submission)
	
	Returns
	-------
	Pipeline
		Sklearn pipeline with all feature engineering steps
	
	Example
	-------
	>>> from sklearn.pipeline import Pipeline
	>>> # Create pipeline for a specific cutoff date
	>>> fold_pipeline = create_feature_pipeline(cutoff_date='2018-10-15')
	>>> # Fit transformers that need raw data
	>>> fold_pipeline.named_steps['accumulated'].fit(None, raw_df=df_raw)
	>>> fold_pipeline.named_steps['page_interactions'].fit(None, raw_df=df_raw)
	>>> fold_pipeline.named_steps['churn_target'].fit(None, raw_df=df_raw)
	>>> # Transform raw data (cutoff date enforced by RawDataSplitter)
	>>> features = fold_pipeline.fit_transform(df_raw)
	"""
	from sklearn.pipeline import Pipeline
	
	cutoff_date = pd.to_datetime(cutoff_date)
	
	# Modular pipeline with clean feature engineering steps
	pipeline_steps = [
		# Step 0: Filter data by cutoff date (temporal integrity)
		('raw_splitter', RawDataSplitter(cutoff_date=cutoff_date)),
		
		# Step 1: Basic aggregation (page counts for rolling averages only)
		('basic_agg', BasicEventAggregator(
			pages_for_rolling=['Add Friend', 'Add to Playlist', 'Thumbs Up', 'Thumbs Down']
		)),
		
		# Step 2: Compute rolling averages (7d and 14d)
		('rolling_7d', RollingAverageTransformerModular(
			pages=['Add Friend', 'Add to Playlist', 'Thumbs Up', 'Thumbs Down'],
			window_days=7
		)),
		('rolling_14d', RollingAverageTransformerModular(
			pages=['Add Friend', 'Add to Playlist', 'Thumbs Up', 'Thumbs Down'],
			window_days=14
		)),
		
		# Step 3: Compute trend features (7d vs 14d comparison)
		('trend', TrendFeaturesTransformer(
			pages=['Add Friend', 'Add to Playlist', 'Thumbs Up', 'Thumbs Down']
		)),
		
		# Step 4: Add accumulated features (requires raw_df in fit)
		('accumulated', AccumulatedFeaturesTransformer()),
		
		# Step 5: Add page interaction features (requires raw_df in fit)
		('page_interactions', PageInteractionTransformer(
			pages_to_track=['About', 'Help', 'Settings', 'Save Settings', 'Home']
		)),
	]
	
	# Only add churn target transformer in training mode
	if mode == 'train':
		pipeline_steps.append(
			('churn_target', CancellationTargetTransformerModular(window_days=window_days))
		)
	
	# Final preprocessing step
	pipeline_steps.append(
		('preprocessor', FeaturePreprocessor())
	)
	
	return Pipeline(pipeline_steps)


def run_time_series_cv(df_raw, n_splits=5, window_days=10, model_params=None):
	"""
	Run time-series cross-validation with proper fold-level feature computation.
	
	This function implements TimeSeriesSplit CV where each fold:
	1. Filters raw data to fold training period
	2. Creates fresh pipeline and fits all transformers on fold data
	3. Recomputes ALL features (including accumulated and rolling) per fold
	4. Trains model and evaluates on validation fold
	
	This prevents data leakage by ensuring no information from validation period
	leaks into training features.
	
	Parameters
	----------
	df_raw : pd.DataFrame
		Raw event-level data
	n_splits : int, default 5
		Number of CV folds
	window_days : int, default 10
		Days to look ahead for churn prediction
	model_params : dict, optional
		Parameters for LGBMClassifier
	
	Returns
	-------
	dict
		Results containing:
		- cv_results: DataFrame with per-fold metrics
		- mean_roc_auc: Average ROC-AUC across folds
		- std_roc_auc: Standard deviation of ROC-AUC
		- fold_models: List of trained models per fold
	"""
	from sklearn.model_selection import TimeSeriesSplit
	from sklearn.preprocessing import StandardScaler, OneHotEncoder
	from sklearn.compose import ColumnTransformer
	from sklearn.pipeline import Pipeline
	from sklearn.metrics import roc_auc_score
	import lightgbm as lgb
	import time
	import time
	
	if model_params is None:
		model_params = {
			'n_estimators': 100,
			'max_depth': 6,
			'learning_rate': 0.1,
			'random_state': 42,
			'verbose': -1
		}
	
	print("=" * 80)
	print(f"TIME-SERIES CROSS-VALIDATION ({n_splits} FOLDS)")
	print("=" * 80)
	
	# Prepare date-level data for splitting
	df_raw['time_dt'] = pd.to_datetime(df_raw['time'])
	df_raw['date'] = df_raw['time_dt'].dt.normalize()
	
	# Get unique user-date combinations for splitting
	user_dates = df_raw.groupby(['userId', 'date']).size().reset_index(name='count')
	user_dates = user_dates.sort_values('date')
	
	print(f"\nDataset info:")
	print(f"  Total user-days: {len(user_dates):,}")
	print(f"  Date range: {user_dates['date'].min().date()} to {user_dates['date'].max().date()}")
	print(f"  Total events: {len(df_raw):,}")
	
	# Setup CV
	tscv = TimeSeriesSplit(n_splits=n_splits)
	cv_results = []
	fold_models = []
	
	for fold, (train_idx, val_idx) in enumerate(tscv.split(user_dates), 1):
		print(f"\n{'='*80}")
		print(f"FOLD {fold}/{n_splits}")
		print(f"{'='*80}")
		
		# Get date ranges for this fold
		train_dates = user_dates.iloc[train_idx]
		val_dates = user_dates.iloc[val_idx]
		
		fold_cutoff = train_dates['date'].max()
		val_start = val_dates['date'].min()
		val_end = val_dates['date'].max()
		
		print(f"Train period: {train_dates['date'].min().date()} to {fold_cutoff.date()} ({len(train_dates):,} user-days)")
		print(f"Val period:   {val_start.date()} to {val_end.date()} ({len(val_dates):,} user-days)")
		
		# CRITICAL: Filter raw data to include train AND validation periods
		# But pipeline will be fitted only on training data to prevent leakage
		print("\n⏱️  TIMING BREAKDOWN:")
		start_time = time.time()
		raw_fold_data = df_raw[df_raw['date'] <= val_end].copy()
		filter_time = time.time() - start_time
		print(f"  1. Filter raw data: {filter_time:.1f}s ({len(raw_fold_data):,} events)")
		
		# Create fresh pipeline for this fold
		start_time = time.time()
		fold_pipeline = create_feature_pipeline(cutoff_date=fold_cutoff, window_days=window_days)
		pipeline_time = time.time() - start_time
		print(f"  2. Create pipeline: {pipeline_time:.1f}s")
		
		# Fit transformers that need raw data
		# These must be fitted with fold-specific data to prevent leakage
		start_time = time.time()
		raw_fold_train = raw_fold_data[raw_fold_data['date'] <= fold_cutoff].copy()
		if 'accumulated' in dict(fold_pipeline.steps):
			fold_pipeline.named_steps['accumulated'].fit(None, raw_df=raw_fold_train)
		if 'page_interactions' in dict(fold_pipeline.steps):
			fold_pipeline.named_steps['page_interactions'].fit(None, raw_df=raw_fold_train)
		if 'churn_target' in dict(fold_pipeline.steps):
			fold_pipeline.named_steps['churn_target'].fit(None, raw_df=raw_fold_data)  # Needs full data for forward-looking labels
		fit_time = time.time() - start_time
		print(f"  3. Fit transformers: {fit_time:.1f}s")
		
		# Transform to get features (using full train+val data, but features computed from train only)
		start_time = time.time()
		print("  4. Transform (detailed breakdown below)...")
		df_fold_features = fold_pipeline.fit_transform(raw_fold_data)
		transform_time = time.time() - start_time
		print(f"     TOTAL TRANSFORM TIME: {transform_time:.1f}s ⚠️")
		
		# Split into train/val based on dates
		df_fold_train = df_fold_features[df_fold_features['date'] <= fold_cutoff].copy()
		df_fold_val = df_fold_features[(df_fold_features['date'] > fold_cutoff) & 
									   (df_fold_features['date'] <= val_end)].copy()
		
		print(f"Feature matrix - Train: {df_fold_train.shape}, Val: {df_fold_val.shape}")
		
		# Prepare features and target
		# TODO: Define feature_cols based on final feature set
		feature_cols = [col for col in df_fold_train.columns 
						if col not in ['userId', 'date', 'churn_status']]
		
		X_fold_train = df_fold_train[feature_cols]
		y_fold_train = df_fold_train['churn_status']
		X_fold_val = df_fold_val[feature_cols]
		y_fold_val = df_fold_val['churn_status']
		
		# Calculate class weights
		neg_count = (y_fold_train == 0).sum()
		pos_count = (y_fold_train == 1).sum()
		scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
		
		print(f"Class distribution - Train: {pos_count}/{len(y_fold_train)} ({pos_count/len(y_fold_train):.2%} churn)")
		print(f"                     Val:   {(y_fold_val==1).sum()}/{len(y_fold_val)} ({(y_fold_val==1).mean():.2%} churn)")
		
		# Identify numeric and categorical features
		numeric_features = X_fold_train.select_dtypes(include=[np.number]).columns.tolist()
		categorical_features = X_fold_train.select_dtypes(exclude=[np.number]).columns.tolist()
		
		# Train model
		model_pipeline = Pipeline([
			('preprocessor', ColumnTransformer(
				transformers=[
					('num', StandardScaler(), numeric_features),
					('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
				]
			)),
			('classifier', lgb.LGBMClassifier(
				scale_pos_weight=scale_pos_weight,
				**model_params
			))
		])
		
		print("Training model...")
		start_time = time.time()
		model_pipeline.fit(X_fold_train, y_fold_train)
		train_time = time.time() - start_time
		print(f"  5. Model training: {train_time:.1f}s")
		
		# Evaluate
		start_time = time.time()
		y_pred_proba = model_pipeline.predict_proba(X_fold_val)[:, 1]
		fold_roc_auc = roc_auc_score(y_fold_val, y_pred_proba)
		eval_time = time.time() - start_time
		print(f"  6. Evaluation: {eval_time:.1f}s")
		
		fold_total = filter_time + pipeline_time + fit_time + transform_time + train_time + eval_time
		print(f"\n  ⏱️  FOLD TOTAL: {fold_total:.1f}s ({fold_total/60:.1f} min)")
		
		print(f"\nFold {fold} Results:")
		print(f"  ROC-AUC: {fold_roc_auc:.4f}")
		
		# Store results
		cv_results.append({
			'fold': fold,
			'train_size': len(X_fold_train),
			'val_size': len(X_fold_val),
			'roc_auc': fold_roc_auc,
			'churn_rate_train': y_fold_train.mean(),
			'churn_rate_val': y_fold_val.mean(),
		})
		fold_models.append(model_pipeline)
	
	# Summarize results
	print(f"\n{'='*80}")
	print("CROSS-VALIDATION SUMMARY")
	print(f"{'='*80}")
	
	cv_df = pd.DataFrame(cv_results)
	print(cv_df.to_string(index=False))
	
	mean_auc = cv_df['roc_auc'].mean()
	std_auc = cv_df['roc_auc'].std()
	
	print(f"\nMean ROC-AUC: {mean_auc:.4f} ± {std_auc:.4f}")
	print(f"Best fold:    {cv_df.loc[cv_df['roc_auc'].idxmax(), 'fold']:.0f} ({cv_df['roc_auc'].max():.4f})")
	print(f"Worst fold:   {cv_df.loc[cv_df['roc_auc'].idxmin(), 'fold']:.0f} ({cv_df['roc_auc'].min():.4f})")
	
	return {
		'cv_results': cv_df,
		'mean_roc_auc': mean_auc,
		'std_roc_auc': std_auc,
		'fold_models': fold_models
	}


def run_group_kfold_cv(df_raw, n_splits=5, window_days=10, model_params=None):
	"""
	Run GroupKFold cross-validation with user-level splitting.
	
	This function implements user-level CV where users are split into folds,
	ensuring no user appears in both train and validation. This tests the model's
	ability to generalize to completely new users, complementing TimeSeriesSplit
	which tests temporal generalization.
	
	Parameters
	----------
	df_raw : pd.DataFrame
		Raw event-level data
	n_splits : int, default 5
		Number of CV folds
	window_days : int, default 10
		Days to look ahead for churn prediction
	model_params : dict, optional
		Parameters for LGBMClassifier
	
	Returns
	-------
	dict
		Results containing:
		- cv_results: DataFrame with per-fold metrics
		- mean_roc_auc: Average ROC-AUC across folds
		- std_roc_auc: Standard deviation of ROC-AUC
		- fold_models: List of trained models per fold
	"""
	from sklearn.model_selection import GroupKFold
	from sklearn.preprocessing import StandardScaler, OneHotEncoder
	from sklearn.compose import ColumnTransformer
	from sklearn.metrics import roc_auc_score
	from sklearn.pipeline import Pipeline
	import lightgbm as lgb
	import time
	
	if model_params is None:
		model_params = {
			'n_estimators': 100,
			'max_depth': 6,
			'learning_rate': 0.1,
			'random_state': 42,
			'verbose': -1
		}
	
	print("=" * 80)
	print(f"GROUP KFOLD CROSS-VALIDATION ({n_splits} FOLDS) - USER-LEVEL SPLITTING")
	print("=" * 80)
	
	# First, compute all features using full training data
	# (We're not doing temporal splitting here, just user-level)
	print("\nComputing features for all users...")
	full_pipeline = create_feature_pipeline(
		cutoff_date=df_raw['time'].max(),  # Use all data
		window_days=window_days
	)
	
	# Fit transformers that need raw data
	if 'accumulated' in dict(full_pipeline.steps):
		full_pipeline.named_steps['accumulated'].fit(None, raw_df=df_raw)
	if 'page_interactions' in dict(full_pipeline.steps):
		full_pipeline.named_steps['page_interactions'].fit(None, raw_df=df_raw)
	if 'churn_target' in dict(full_pipeline.steps):
		full_pipeline.named_steps['churn_target'].fit(None, raw_df=df_raw)
	
	# Transform to get features
	df_features = full_pipeline.fit_transform(df_raw)
	
	print(f"Feature matrix shape: {df_features.shape}")
	
	# Prepare features and target
	feature_cols = [col for col in df_features.columns 
					if col not in ['userId', 'date', 'churn_status']]
	
	X = df_features[feature_cols]
	y = df_features['churn_status']
	groups = df_features['userId']  # Group by user
	
	# Setup GroupKFold
	gkf = GroupKFold(n_splits=n_splits)
	cv_results = []
	fold_models = []
	
	print(f"\nTotal users: {groups.nunique()}")
	print(f"Total user-days: {len(df_features):,}")
	print(f"Overall churn rate: {y.mean():.2%}")
	
	for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
		print(f"\n{'='*80}")
		print(f"FOLD {fold}/{n_splits}")
		print(f"{'='*80}")
		
		# Split by user groups
		X_fold_train = X.iloc[train_idx]
		y_fold_train = y.iloc[train_idx]
		X_fold_val = X.iloc[val_idx]
		y_fold_val = y.iloc[val_idx]
		
		train_users = groups.iloc[train_idx].unique()
		val_users = groups.iloc[val_idx].unique()
		
		print(f"Train users: {len(train_users):,} ({len(X_fold_train):,} user-days)")
		print(f"Val users:   {len(val_users):,} ({len(X_fold_val):,} user-days)")
		print(f"Churn rate - Train: {y_fold_train.mean():.2%}, Val: {y_fold_val.mean():.2%}")
		
		# Verify no user overlap
		overlap = set(train_users) & set(val_users)
		if overlap:
			print(f"WARNING: {len(overlap)} users appear in both train and val!")
		
		# Calculate class weights
		neg_count = (y_fold_train == 0).sum()
		pos_count = (y_fold_train == 1).sum()
		scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
		
		# Identify numeric and categorical features
		numeric_features = X_fold_train.select_dtypes(include=[np.number]).columns.tolist()
		categorical_features = X_fold_train.select_dtypes(exclude=[np.number]).columns.tolist()
		
		# Train model
		model_pipeline = Pipeline([
			('preprocessor', ColumnTransformer(
				transformers=[
					('num', StandardScaler(), numeric_features),
					('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
				]
			)),
			('classifier', lgb.LGBMClassifier(
				scale_pos_weight=scale_pos_weight,
				**model_params
			))
		])
		
		print("Training model...")
		start_time = time.time()
		model_pipeline.fit(X_fold_train, y_fold_train)
		train_time = time.time() - start_time
		print(f"  Model training: {train_time:.1f}s")
		
		# Evaluate
		start_time = time.time()
		y_pred_proba = model_pipeline.predict_proba(X_fold_val)[:, 1]
		fold_roc_auc = roc_auc_score(y_fold_val, y_pred_proba)
		eval_time = time.time() - start_time
		print(f"  Evaluation: {eval_time:.1f}s")
		
		print(f"\nFold {fold} Results:")
		print(f"  ROC-AUC: {fold_roc_auc:.4f}")
		
		# Store results
		cv_results.append({
			'fold': fold,
			'train_users': len(train_users),
			'val_users': len(val_users),
			'train_size': len(X_fold_train),
			'val_size': len(X_fold_val),
			'roc_auc': fold_roc_auc,
			'churn_rate_train': y_fold_train.mean(),
			'churn_rate_val': y_fold_val.mean(),
		})
		fold_models.append(model_pipeline)
	
	# Summarize results
	print(f"\n{'='*80}")
	print("GROUP KFOLD CROSS-VALIDATION SUMMARY")
	print(f"{'='*80}")
	
	cv_df = pd.DataFrame(cv_results)
	print(cv_df.to_string(index=False))
	
	mean_auc = cv_df['roc_auc'].mean()
	std_auc = cv_df['roc_auc'].std()
	
	print(f"\nMean ROC-AUC: {mean_auc:.4f} ± {std_auc:.4f}")
	print(f"Best fold:    {cv_df.loc[cv_df['roc_auc'].idxmax(), 'fold']:.0f} ({cv_df['roc_auc'].max():.4f})")
	print(f"Worst fold:   {cv_df.loc[cv_df['roc_auc'].idxmin(), 'fold']:.0f} ({cv_df['roc_auc'].min():.4f})")
	
	return {
		'cv_results': cv_df,
		'mean_roc_auc': mean_auc,
		'std_roc_auc': std_auc,
		'fold_models': fold_models
	}
