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
	
	# Get user registration dates (first observed timestamp per user)
	user_registration = df_copy.groupby(user_col)[time_col].min().reset_index()
	user_registration.columns = [user_col, 'registration_date']
	user_registration['registration_date'] = pd.to_datetime(user_registration['registration_date']).dt.normalize()
	
	# Get last level observation per user per day
	if level_col in df_copy.columns:
		level_per_day = df_copy.groupby([user_col, 'date'])[level_col].last().reset_index()
	else:
		print(f"Warning: '{level_col}' column not found, skipping level tracking")
		level_per_day = None
	
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
	
	# Add level column with forward-filling
	if level_per_day is not None:
		df_aggregated = df_aggregated.merge(level_per_day, on=[user_col, 'date'], how='left')
		
		# Forward-fill level per user
		df_aggregated['date_temp'] = pd.to_datetime(df_aggregated['date'])
		df_aggregated = df_aggregated.sort_values([user_col, 'date_temp'])
		df_aggregated['level'] = df_aggregated.groupby(user_col)['level'].ffill()
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
		
		# Compute rolling averages per column using prior days only (NO LEAKAGE)
		import numpy as np
		for col in columns:
			col_name_lower = col.lower().replace(' ', '_')
			new_col_name = f'{col_name_lower}_avg_{n}d'
			
			# FIX: Use explicit date-based windowing instead of shift() + rolling()
			# This ensures we ONLY look at data strictly BEFORE the current date
			rolling_avgs = []
			for idx, row in user_data.iterrows():
				current_date = row[date_col]
				window_start = current_date - pd.Timedelta(days=n)
				
				# Select data strictly BEFORE current_date (not including it)
				window_data = user_data[
					(user_data[date_col] > window_start) & 
					(user_data[date_col] < current_date)  # <- EXCLUDE current date
				]
				
				if len(window_data) > 0:
					avg_val = window_data[col].mean()
				else:
					avg_val = np.nan
				
				rolling_avgs.append(avg_val)
			
			user_data[new_col_name] = rolling_avgs
		
		processed_users.append(user_data)

	# Combine all users and restore date to date type for consistency
	result = pd.concat(processed_users, ignore_index=True)
	result[date_col] = result[date_col].dt.date

	return result