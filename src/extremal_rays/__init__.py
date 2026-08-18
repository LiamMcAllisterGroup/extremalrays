"""Fast extremal rays of pointed polyhedral cones via Clarkson's algorithm."""

from .core import extremal_rays, positive_functional
from .verify import verify_extremal_rays

__version__ = "0.1.0"
__all__ = ["extremal_rays", "positive_functional", "verify_extremal_rays"]
