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
                                  drop_cancellation: bool = True) -> pd.DataFrame:
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
	
	Returns
	-------
	pd.DataFrame
		Aggregated DataFrame with columns:
		- user_col: User identifier
		- 'date': Date extracted from time_col
		- One column per unique page category with count values
	
	Raises
	------
	ValueError
		If required columns are not found in the DataFrame
	
	Example
	-------
	>>> df_aggregated = aggregate_user_day_activity(df)
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
	
	# Extract the date from the time column
	df_copy['date'] = df_copy[time_col].dt.date
	
	# Create pivot table with counts for each page category per user per day
	df_aggregated = df_copy.groupby([user_col, 'date', page_col]).size().unstack(fill_value=0).reset_index()
	
	# Optionally drop the cancellation confirmation column
	if drop_cancellation and 'Cancellation Confirmation' in df_aggregated.columns:
		df_aggregated = df_aggregated.drop(columns=['Cancellation Confirmation'])
	
	return df_aggregated


def add_days_since(df: pd.DataFrame,
                           columns_to_track: list = None,
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
	columns_to_track : list, default ['Submit Downgrade', 'Submit Upgrade', 'Cancel']
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
	...     columns_to_track=['Submit Downgrade', 'Submit Upgrade', 'Cancel'])
	>>> print(df_agg.columns)
	['userId', 'date', ..., 'days_since_submit_downgrade', 
	 'days_since_submit_upgrade', 'days_since_cancel']
	"""
	if columns_to_track is None:
		columns_to_track = ['Submit Downgrade', 'Submit Upgrade', 'Cancel']
	
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
                         date_col: str = 'date') -> pd.DataFrame:
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
	
	# Ensure date column is in date format
	if not isinstance(df_copy[date_col].iloc[0], (pd.Timestamp, type(pd.Timestamp('2020-01-01').date()))):
		df_copy[date_col] = pd.to_datetime(df_copy[date_col])
		if hasattr(df_copy[date_col].iloc[0], 'date'):
			df_copy[date_col] = df_copy[date_col].dt.date
	
	df_copy = df_copy.sort_values([user_col, date_col])
	
	# Process each column
	for col in columns:
		col_name_lower = col.lower().replace(' ', '_')
		new_col_name = f'{col_name_lower}_avg_{n}d'
		
		# Initialize the new column with NaN
		df_copy[new_col_name] = None
		
		# For each user, calculate rolling averages
		result_list = []
		for user_id in df_copy[user_col].unique():
			user_data = df_copy[df_copy[user_col] == user_id].copy()
			
			# For each row, calculate the rolling average for the n days before that row's date
			for idx, row in user_data.iterrows():
				current_date = row[date_col]
				window_start = current_date - pd.Timedelta(days=n)
				
				# Get the user's historical data in the window (n days before current date, not including current date)
				user_history = user_data[
					(user_data[date_col] > window_start) & 
					(user_data[date_col] < current_date)
				].copy()
				
				if len(user_history) > 0:
					# Calculate the average for this user over the window
					avg_value = user_history[col].mean()
					user_data.loc[idx, new_col_name] = avg_value
			
			result_list.append(user_data)
		
		df_copy = pd.concat(result_list, ignore_index=True)
	
	return df_copy