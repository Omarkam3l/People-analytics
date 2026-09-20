"""Comparative analysis between image-space and ground-space analytics.

Note:
    Comparison between image-space and ARBITRARY_PLANAR analytics is strictly a
    coordinate-representation comparison. It must NOT be interpreted as demonstrating
    higher accuracy, physical correctness, or metric improvement, because Phase 10
    controlled manual calibration does not provide physical metric ground truth.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from people_analytics.dwell.models import ZoneVisit
from people_analytics.ground_analytics.models import GroundZoneMembership
from people_analytics.zone.models import ZoneMembership


@dataclass(frozen=True)
class GroundComparisonReport:
    """Quantitative comparison between image-space and ground-space analytics.

    Represents differences resulting from planar coordinate transformation without
    implying physical metric superiority or ground-truth calibration accuracy.
    """
    image_visits_total: int
    ground_visits_total: int
    image_unique_visitors: int
    ground_unique_visitors: int
    image_dwell_seconds_total: float
    ground_dwell_seconds_total: float
    visits_by_zone_image: Dict[str, int]
    visits_by_zone_ground: Dict[str, int]
    membership_agreement_count: int
    membership_total_evaluated: int
    agreement_ratio: float


def compare_image_and_ground_analytics(
    image_memberships: Sequence[ZoneMembership],
    ground_memberships: Sequence[GroundZoneMembership],
    image_visits: Sequence[ZoneVisit],
    ground_visits: Sequence[ZoneVisit],
    zones_to_compare: Optional[Sequence[str]] = None,
) -> GroundComparisonReport:
    """Compare image-space and ground-space memberships and dwell visits.

    This evaluates cross-representation consistency between image pixel space
    and the ARBITRARY_PLANAR coordinate representation.

    Args:
        image_memberships: Sequence of ZoneMembership observations from Phase 7.
        ground_memberships: Sequence of GroundZoneMembership observations from Phase 11.
        image_visits: Sequence of finalized ZoneVisit records from Phase 8.
        ground_visits: Sequence of finalized ZoneVisit records from Phase 11.
        zones_to_compare: Optional subset of zone IDs to check for agreement.

    Returns:
        GroundComparisonReport instance with detailed agreement metrics.
    """
    # Zone visit tallies
    img_visits_by_zone: Dict[str, int] = {}
    for v in image_visits:
        img_visits_by_zone[v.zone_id] = img_visits_by_zone.get(v.zone_id, 0) + 1

    grd_visits_by_zone: Dict[str, int] = {}
    for v in ground_visits:
        grd_visits_by_zone[v.zone_id] = grd_visits_by_zone.get(v.zone_id, 0) + 1

    # Membership agreement across aligned observations
    agreement_count = 0
    total_evaluated = min(len(image_memberships), len(ground_memberships))

    for i in range(total_evaluated):
        m_img = image_memberships[i]
        m_grd = ground_memberships[i]
        if m_img.track_id == m_grd.track_id and m_img.frame_index == m_grd.frame_index:
            set_img = set(m_img.zone_ids)
            set_grd = set(m_grd.zone_ids)
            if zones_to_compare is not None:
                set_img &= set(zones_to_compare)
                set_grd &= set(zones_to_compare)
            if set_img == set_grd:
                agreement_count += 1

    ratio = agreement_count / total_evaluated if total_evaluated > 0 else 1.0

    return GroundComparisonReport(
        image_visits_total=len(image_visits),
        ground_visits_total=len(ground_visits),
        image_unique_visitors=len(set(v.track_id for v in image_visits)),
        ground_unique_visitors=len(set(v.track_id for v in ground_visits)),
        image_dwell_seconds_total=sum(v.duration_seconds for v in image_visits),
        ground_dwell_seconds_total=sum(v.duration_seconds for v in ground_visits),
        visits_by_zone_image=img_visits_by_zone,
        visits_by_zone_ground=grd_visits_by_zone,
        membership_agreement_count=agreement_count,
        membership_total_evaluated=total_evaluated,
        agreement_ratio=ratio,
    )
