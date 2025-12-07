import pandas as pd
from typing import Union

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


def aggregate_user_day_activity(df: pd.DataFrame,
                                  user_col: str = 'userId',
                                  time_col: str = 'time',
                                  page_col: str = 'page',
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
	
	# Create pivot table with counts for each page category per user per day
	df_aggregated = df_copy.groupby([user_col, 'date', page_col]).size().unstack(fill_value=0).reset_index()
	
	# Optionally drop the cancellation confirmation column
	if drop_cancellation and 'Cancellation Confirmation' in df_aggregated.columns:
		df_aggregated = df_aggregated.drop(columns=['Cancellation Confirmation'])
	
	# Fill missing days with zeros per user if requested
	if fill_missing_days:
		activity_cols = [col for col in df_aggregated.columns if col not in [user_col, 'date']]
		filled_users = []
		
		for user_id, user_data in df_aggregated.groupby(user_col):
			# Create full date range for this user
			full_range = pd.date_range(user_data['date'].min(), user_data['date'].max(), freq='D')
			
			# Reindex to include all days
			user_filled = user_data.set_index('date').reindex(full_range)
			user_filled[user_col] = user_id
			
			# Fill activity columns with zeros for inactive days
			user_filled[activity_cols] = user_filled[activity_cols].fillna(0).astype(int)
			
			user_filled.index.name = 'date'
			filled_users.append(user_filled.reset_index())
		
		df_aggregated = pd.concat(filled_users, ignore_index=True)
	
	# Convert date back to date type (not datetime) for consistency
	df_aggregated['date'] = df_aggregated['date'].dt.date
	
	return df_aggregated


def add_days_since(df: pd.DataFrame,
                           columns_to_track: list = ['Submit Downgrade', 'Submit Upgrade'],
                           user_col: str = 'userId',
                           date_col: str = 'date') -> pd.DataFrame:
	"""
	Add "days since" columns for specified event columns in an aggregated user-day DataFrame.
	
	For each specified column, creates a new column named 'days_since_<col_name_lowercase>'
	that tracks the number of days since the user's most recent occurrence of that event
	(up to and including the current date).
	
	Parameters
	----------
	df : pd.DataFrame
		Aggregated user-day DataFrame (typically output from aggregate_user_day_activity)
	columns_to_track : list, default ['Submit Downgrade', 'Submit Upgrade']
		List of column names to create "days since" tracking for
	user_col : str, default 'userId'
		Column name identifying the user
	date_col : str, default 'date'
		Column name containing the date values
	
	Returns
	-------
	pd.DataFrame
		DataFrame with new 'days_since_<col>' columns added for each tracked event
	
	Raises
	------
	ValueError
		If required columns are not found in the DataFrame
	
	Example
	-------
	>>> df_agg = add_days_since(df_agg, 
	...     columns_to_track=['Submit Downgrade', 'Submit Upgrade'])
	>>> print(df_agg.columns)
	['userId', 'date', ..., 'days_since_submit_downgrade', 
	 'days_since_submit_upgrade']
	"""

	# Validate required columns
	if user_col not in df.columns:
		raise ValueError(f"user_col '{user_col}' not found in DataFrame")
	if date_col not in df.columns:
		raise ValueError(f"date_col '{date_col}' not found in DataFrame")
	
	for col in columns_to_track:
		if col not in df.columns:
			raise ValueError(f"Column '{col}' not found in DataFrame")
	
	# Work with a copy to avoid modifying the original
	df_copy = df.copy()
	df_copy = df_copy.sort_values([user_col, date_col])
	
	# Process each column
	for col in columns_to_track:
		col_name_lower = col.lower().replace(' ', '_')
		new_col_name = f'days_since_{col_name_lower}'
		
		# Identify dates where the event occurred for each user
		df_with_event = df_copy[df_copy[col] > 0].copy()
		df_with_event = df_with_event[[user_col, date_col]].rename(columns={date_col: f'last_{col_name_lower}_date'})
		
		# Initialize the new column with None
		df_copy[new_col_name] = None
		
		# For each user, calculate days since their last event
		result_list = []
		for user_id in df_copy[user_col].unique():
			user_data = df_copy[df_copy[user_col] == user_id].copy()
			event_dates = df_with_event[df_with_event[user_col] == user_id][f'last_{col_name_lower}_date'].values
			
			if len(event_dates) > 0:
				for idx, row in user_data.iterrows():
					# Find the most recent event date before or on the current date
					prior_events = event_dates[event_dates <= row[date_col]]
					if len(prior_events) > 0:
						last_event = prior_events.max()
						days_since = (row[date_col] - last_event).days
						user_data.loc[idx, new_col_name] = days_since
			
			result_list.append(user_data)
		
		df_copy = pd.concat(result_list, ignore_index=True)
	
	return df_copy


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

	# Normalize date column to datetime (no time component)
	df_copy[date_col] = pd.to_datetime(df_copy[date_col]).dt.normalize()
	
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
			# Ensure index is clean for rolling operations without reindexing
			user_data = user_data.reset_index(drop=True)
		
		# Compute rolling averages per column using prior days only
		for col in columns:
			col_name_lower = col.lower().replace(' ', '_')
			new_col_name = f'{col_name_lower}_avg_{n}d'
			# shift(1) excludes the current day from the window
			user_data[new_col_name] = user_data[col].shift(1).rolling(window=n, min_periods=1).mean()
		
		processed_users.append(user_data)

	# Combine all users and restore date to date type for consistency
	result = pd.concat(processed_users, ignore_index=True)
	result[date_col] = result[date_col].dt.date

	return result


def add_thumbs_ratio(df: pd.DataFrame, 
                     thumbs_up_col: str = 'Thumbs Up',
                     thumbs_down_col: str = 'Thumbs Down') -> pd.DataFrame:
	"""
	Add thumbs up to thumbs down ratio feature.
	
	Calculates the ratio of thumbs up to total thumbs interactions (up + down).
	Handles division by zero by setting ratio to 0.5 when no thumbs interactions exist.
	
	Parameters
	----------
	df : pd.DataFrame
		Input dataframe with thumbs up/down activity
	thumbs_up_col : str
		Column name for thumbs up counts
	thumbs_down_col : str
		Column name for thumbs down counts
	
	Returns
	-------
	pd.DataFrame
		Input dataframe with new 'thumbs_ratio' column added
	"""
	df = df.copy()
	
	# Calculate total thumbs interactions
	total_thumbs = df[thumbs_up_col] + df[thumbs_down_col]
	
	# Calculate ratio: thumbs_up / (thumbs_up + thumbs_down)
	# When no thumbs interactions, set to 0.5 (neutral)
	df['thumbs_ratio'] = df[thumbs_up_col] / total_thumbs
	df['thumbs_ratio'] = df['thumbs_ratio'].fillna(0.5)
	
	# Handle any remaining NaN or inf values
	df['thumbs_ratio'] = df['thumbs_ratio'].replace([float('inf'), float('-inf')], 0.5)
	
	print(f"Added thumbs_ratio feature")
	print(f"thumbs_ratio range: [{df['thumbs_ratio'].min():.2f}, {df['thumbs_ratio'].max():.2f}]")
	
	return df


def add_days_active_last_n_days(df: pd.DataFrame,
                                user_col: str = 'userId',
                                date_col: str = 'date',
                                n_days: int = 30,
                                activity_threshold: float = 0) -> pd.DataFrame:
	"""
	Add feature counting days with activity in last n days.
	
	For each user on each date, counts how many unique days in the preceding n days
	had activity (where activity is any row with activity > activity_threshold).
	
	Parameters
	----------
	df : pd.DataFrame
		Input dataframe with user activity
	user_col : str
		Column identifying users
	date_col : str
		Column containing dates
	n_days : int
		Number of days to look back (default: 30)
	activity_threshold : float
		Minimum activity level to count as "active day" (default: 0)
	
	Returns
	-------
	pd.DataFrame
		Input dataframe with new 'days_active_last_Xd' column added
	"""
	df = df.copy()
	
	# Ensure date column is datetime for calculations
	df[date_col] = pd.to_datetime(df[date_col])
	
	processed_users = []
	
	for user_id in df[user_col].unique():
		user_data = df[df[user_col] == user_id].copy()
		user_data = user_data.sort_values(date_col).reset_index(drop=True)
		
		days_active = []
		
		for idx, row in user_data.iterrows():
			current_date = row[date_col]
			window_start = current_date - pd.Timedelta(days=n_days-1)
			
			# Count unique days with activity in the window
			# (including the current day)
			window_data = user_data[
				(user_data[date_col] >= window_start) & 
				(user_data[date_col] <= current_date)
			]
			
			# Count days where user had any activity
			# Sum across all activity columns (numeric columns except identifiers)
			activity_cols = [col for col in window_data.columns 
			                 if col not in [user_col, date_col] 
			                 and window_data[col].dtype in ['int64', 'float64']]
			
			if activity_cols:
				total_activity_per_day = window_data[activity_cols].sum(axis=1)
				days_with_activity = (total_activity_per_day > activity_threshold).sum()
			else:
				days_with_activity = 0
			
			days_active.append(days_with_activity)
		
		user_data[f'days_active_last_{n_days}d'] = days_active
		processed_users.append(user_data)
	
	result = pd.concat(processed_users, ignore_index=True)
	
	# Convert date back to date type if it was originally
	if result[date_col].dtype != 'object':
		result[date_col] = result[date_col].dt.date
	
	print(f"Added days_active_last_{n_days}d feature")
	print(f"days_active_last_{n_days}d range: [{result[f'days_active_last_{n_days}d'].min()}, {result[f'days_active_last_{n_days}d'].max()}]")
	
	return result
