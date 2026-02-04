"""
Global constant definitions for platforms and corresponding API base URLs.
"""
from typing import Literal

# Define the type of the platform
PLATFORM = Literal['openai', 'google']

# Define the base URL for each platform
BASE_URL_MAP = {
    'openai': None,
    'google': 'https://generativelanguage.googleapis.com/v1beta'
}