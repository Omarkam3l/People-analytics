"""Pure geometric algorithms for polygon validation and point-in-polygon queries."""

import math
from typing import Sequence, Tuple


def is_point_on_segment(
    pt: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    tol: float = 1e-7,
) -> bool:
    """Check if point pt lies on line segment (p1, p2) within distance tolerance."""
    px, py = pt
    x1, y1 = p1
    x2, y2 = p2

    # Check bounding box of segment first with tolerance
    if px < min(x1, x2) - tol or px > max(x1, x2) + tol:
        return False
    if py < min(y1, y2) - tol or py > max(y1, y2) + tol:
        return False

    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy

    if seg_len_sq <= tol * tol:
        # p1 and p2 are essentially identical
        dist_sq = (px - x1) * (px - x1) + (py - y1) * (py - y1)
        return dist_sq <= tol * tol

    # Cross product: |(p2 - p1) x (p1 - pt)| / |p2 - p1|
    cross = abs((x2 - x1) * (y1 - py) - (x1 - px) * (y2 - y1))
    dist = cross / math.sqrt(seg_len_sq)
    return dist <= tol


def segments_intersect(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    p4: Tuple[float, float],
) -> bool:
    """Determine if line segments (p1, p2) and (p3, p4) properly intersect."""
    def ccw(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])

    d1 = ccw(p1, p2, p3)
    d2 = ccw(p1, p2, p4)
    d3 = ccw(p3, p4, p1)
    d4 = ccw(p3, p4, p2)

    # Proper intersection requires strict opposite signs
    if ((d1 > 1e-9 and d2 < -1e-9) or (d1 < -1e-9 and d2 > 1e-9)) and \
       ((d3 > 1e-9 and d4 < -1e-9) or (d3 < -1e-9 and d4 > 1e-9)):
        return True

    return False


def compute_polygon_area(vertices: Sequence[Tuple[float, float]]) -> float:
    """Compute the absolute enclosed area of a polygon using the shoelace formula."""
    n = len(vertices)
    if n < 3:
        return 0.0

    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]

    return abs(area) / 2.0


def validate_polygon_vertices(
    raw_vertices: Sequence[Tuple[float, float]],
) -> Tuple[Tuple[float, float], ...]:
    """Validate and normalize polygon vertices.

    Rules:
        - All coordinates must be finite real numbers (no NaN or Inf).
        - Explicit closing duplicate vertex (v[-1] == v[0]) is stripped.
        - Must contain at least 3 unique vertices.
        - Consecutive duplicate vertices are forbidden.
        - Enclosed area must be strictly positive (shoelace area > 1e-6).
        - Non-adjacent edges must not intersect (rejects self-intersecting polygons like bow-ties).

    Returns:
        Immutable tuple of normalized vertices.

    Raises:
        ValueError: If any validation rule is violated.
    """
    if len(raw_vertices) < 3:
        raise ValueError(f"Polygon must have at least 3 vertices, got {len(raw_vertices)}")

    # Check finite numbers
    for i, pt in enumerate(raw_vertices):
        if len(pt) != 2:
            raise ValueError(f"Vertex at index {i} must be a 2D coordinate (x, y), got {pt}")
        x, y = pt
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"Vertex at index {i} contains non-finite coordinate: {pt}")

    # Strip explicit closing duplicate if provided
    verts = list(raw_vertices)
    if len(verts) > 3 and verts[0] == verts[-1]:
        verts.pop()

    n = len(verts)
    if n < 3:
        raise ValueError(f"Polygon must have at least 3 unique vertices, got {n}")

    # Check consecutive duplicates
    for i in range(n):
        next_i = (i + 1) % n
        if verts[i] == verts[next_i]:
            raise ValueError(f"Polygon contains consecutive duplicate vertex at index {i}: {verts[i]}")

    # Check non-adjacent edge intersections (self-intersection test)
    for i in range(n):
        p1 = verts[i]
        p2 = verts[(i + 1) % n]
        for j in range(i + 2, n):
            if (i == 0 and j == n - 1):
                continue  # adjacent via polygon wrap-around
            p3 = verts[j]
            p4 = verts[(j + 1) % n]
            if segments_intersect(p1, p2, p3, p4):
                raise ValueError(
                    f"Polygon is self-intersecting: edge ({p1}, {p2}) intersects edge ({p3}, {p4})"
                )

    # Check non-degenerate area
    area = compute_polygon_area(verts)
    if area <= 1e-6:
        raise ValueError(f"Polygon has degenerate or zero area: {area:.8f}")

    return tuple((float(x), float(y)) for x, y in verts)


def is_point_in_polygon(
    pt: Tuple[float, float],
    vertices: Sequence[Tuple[float, float]],
    inclusive: bool = True,
    tol: float = 1e-7,
) -> bool:
    """Test if point pt lies inside polygon using closed-boundary inclusive ray-casting.

    Semantics:
        - Points strictly inside evaluate to True.
        - Points strictly outside evaluate to False.
        - Points lying on an edge or vertex evaluate to True when inclusive=True.

    Args:
        pt: Coordinate (x, y).
        vertices: Sequence of polygon vertices in order.
        inclusive: Whether boundary points (edges and vertices) are considered inside (default: True).
        tol: Floating-point tolerance for edge distance check.

    Returns:
        True if point is within the polygon, False otherwise.
    """
    n = len(vertices)
    if n < 3:
        return False

    px, py = pt

    # 1. Closed-boundary check: test if point lies directly on any boundary segment
    if inclusive:
        for i in range(n):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % n]
            if is_point_on_segment(pt, p1, p2, tol=tol):
                return True

    # 2. Standard ray-casting (crossing number) for interior testing
    # Cast a horizontal ray to the right: (px, py) to (+inf, py)
    inside = False
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]

        # Check if horizontal ray crosses segment (p1, p2)
        if (y1 > py) != (y2 > py):
            # Compute x-coordinate of intersection
            x_intersect = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if px < x_intersect:
                inside = not inside

    return inside
