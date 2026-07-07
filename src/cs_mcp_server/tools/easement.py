# Copyright contributors to the IBM Core Content Services MCP Server project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Easement mapping tools: place a survey traverse from a recorded easement
document onto real-world coordinates using county parcel and PLSS GIS data."""

import base64
import gzip
import json
import logging
import math
import re
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import quote

import aiohttp
from fastmcp import FastMCP
from mcp.types import ToolAnnotations

from cs_mcp_server.utils.common import ToolError

logger = logging.getLogger(__name__)

FT_PER_M = 3.28083333

# County GIS adapter registry. Each adapter defines the ArcGIS REST layers and
# field names used for parcel and PLSS-section lookups. Add counties as needed.
COUNTY_GIS: Dict[str, Dict[str, str]] = {
    "maricopa": {
        "parcel_url": (
            "https://gis.maricopa.gov/arcgis/rest/services/IndividualService/"
            "Parcel/MapServer/1/query"
        ),
        "apn_field": "APNDash",
        "owner_field": "OwnerName",
        "address_field": "PropertyFullStreetAddress",
        "subdivision_field": "SubdivisionName",
        "lot_field": "Lot",
        # PLSS sections: Township='T1S', Range='R7E', Section='31' (strings);
        # QuarterSection='' selects the full-section polygon.
        "trs_url": (
            "https://gis.maricopa.gov/arcgis/rest/services/IndividualService/"
            "TownshipRangeSection/MapServer/3/query"
        ),
    },
}


def _parse_bearing(bearing: str, dist_ft: float) -> Tuple[float, float]:
    """Quadrant bearing (e.g. N89°51'34"E, N89-51-34E, S00 07 32 E) + distance
    -> (dx_east_ft, dy_north_ft)."""
    m = re.match(
        r"^\s*([NS])\s*(\d{1,3})\D+(\d{1,2})\D+([\d.]+)\D*([EW])\s*$",
        bearing.upper(),
    )
    if not m:
        raise ValueError(f"Cannot parse bearing: {bearing!r}")
    ns, deg, minutes, seconds, ew = m.groups()
    theta = math.radians(float(deg) + float(minutes) / 60 + float(seconds) / 3600)
    dy = math.cos(theta) * dist_ft * (1 if ns == "N" else -1)
    dx = math.sin(theta) * dist_ft * (1 if ew == "E" else -1)
    return dx, dy


def _meters_per_degree(lat_deg: float) -> Tuple[float, float]:
    phi = math.radians(lat_deg)
    m_lat = 111132.92 - 559.82 * math.cos(2 * phi) + 1.175 * math.cos(4 * phi)
    m_lon = 111412.84 * math.cos(phi) - 93.5 * math.cos(3 * phi)
    return m_lat, m_lon


def _edge_metrics(ring: List[List[float]]) -> List[Dict[str, Any]]:
    """Per-edge midpoint and length in meters for a lon/lat ring."""
    edges = []
    for i in range(len(ring) - 1):
        (lon1, lat1), (lon2, lat2) = ring[i], ring[i + 1]
        m_lat, m_lon = _meters_per_degree((lat1 + lat2) / 2)
        length_m = math.hypot((lon2 - lon1) * m_lon, (lat2 - lat1) * m_lat)
        edges.append(
            {
                "i": i,
                "p1": ring[i],
                "p2": ring[i + 1],
                "mid_lat": (lat1 + lat2) / 2,
                "mid_lon": (lon1 + lon2) / 2,
                "length_m": length_m,
            }
        )
    return edges


def _pick_corner(ring: List[List[float]], corner: str) -> Optional[List[float]]:
    """Vertex of a lon/lat ring best matching a named corner (NW/NE/SW/SE)."""
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    lo_min, lo_max = min(lons), max(lons)
    la_min, la_max = min(lats), max(lats)
    d_lo = (lo_max - lo_min) or 1e-12
    d_la = (la_max - la_min) or 1e-12
    corner = corner.strip().upper()
    if corner not in ("NW", "NE", "SW", "SE"):
        return None
    best, best_score = None, -9.0
    for lon, lat in ring:
        u = (lon - lo_min) / d_lo  # 0 = west, 1 = east
        v = (lat - la_min) / d_la  # 0 = south, 1 = north
        score = {
            "NW": (1 - u) + v,
            "NE": u + v,
            "SW": (1 - u) + (1 - v),
            "SE": u + (1 - v),
        }[corner]
        if score > best_score:
            best_score, best = score, [lon, lat]
    return best


def _point_in_ring(lon: float, lat: float, ring: List[List[float]]) -> bool:
    """Even-odd ray casting point-in-polygon test."""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > lat) != (yj > lat):
            x_cross = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < x_cross:
                inside = not inside
        j = i
    return inside


def _containment_score(
    verts_local: List[Tuple[float, float]],
    pob_lon: float,
    pob_lat: float,
    parcel_ring: List[List[float]],
) -> float:
    """Fraction of the placed easement's test points lying inside the parcel.

    Test points are the traverse vertices plus edge midpoints, nudged slightly
    toward the easement's own centroid so on-boundary points count as inside.
    """
    m_lat, m_lon = _meters_per_degree(pob_lat)
    pts = []
    for i in range(len(verts_local) - 1):
        x1, y1 = verts_local[i]
        x2, y2 = verts_local[i + 1]
        pts.append((x1, y1))
        pts.append(((x1 + x2) / 2, (y1 + y2) / 2))
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    inside = 0
    for x_ft, y_ft in pts:
        # nudge ~1 ft toward the centroid to keep boundary points unambiguous
        dxc, dyc = cx - x_ft, cy - y_ft
        norm = math.hypot(dxc, dyc) or 1.0
        xt = x_ft + dxc / norm
        yt = y_ft + dyc / norm
        lon = pob_lon + (xt / FT_PER_M) / m_lon
        lat = pob_lat + (yt / FT_PER_M) / m_lat
        if _point_in_ring(lon, lat, parcel_ring):
            inside += 1
    return inside / len(pts)


def _parse_course_rows(rows_json: str, what: str) -> List[Tuple[str, str, float]]:
    """Parse a JSON array of course rows into (label, bearing, distance_ft)."""
    rows = json.loads(rows_json)
    return [
        (
            str(row.get("line", f"{what}{i + 1}")),
            str(row["bearing"]),
            float(row.get("distance_ft", row.get("distance"))),
        )
        for i, row in enumerate(rows)
    ]


async def _gis_query(url: str) -> Dict[str, Any]:
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status != 200:
                raise RuntimeError(f"GIS query failed with HTTP {resp.status}")
            return await resp.json(content_type=None)


def register_easement_tools(mcp: FastMCP) -> None:
    @mcp.tool(
        name="map_easement_to_parcel",
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def map_easement_to_parcel(
        line_table_json: str,
        apn: str = "",
        address: str = "",
        subdivision: str = "",
        lot: str = "",
        county: str = "maricopa",
        pob_anchor: str = "parcel_edge",
        pob_edge: str = "north",
        pob_from_end: str = "east",
        pob_from_corner_ft: float = 0.0,
        pob_corner: str = "",
        section: str = "",
        township: str = "",
        range_: str = "",
        tie_courses_json: str = "",
    ) -> Union[Dict[str, Any], ToolError]:
        """
        Places an easement survey traverse onto real-world map coordinates and
        returns GeoJSON. The traverse (from the survey's line table or prose
        metes-and-bounds courses) is chained and closure-checked, then anchored
        to the ground using county parcel and/or PLSS section GIS data.

        This tool performs no repository operations — it is a pure geospatial
        computation using public county GIS data.

        :param line_table_json: JSON array of the traverse courses in order, e.g.
            '[{"line": "L1", "bearing": "S00-07-32E", "distance_ft": 10.30}, ...]'.
            Bearings accept degree symbols or dashes (N89°51'34"E or N89-51-34E).
        :param apn: Assessor's Parcel Number as printed (e.g. "200-18-001S"). Optional if
            address or subdivision+lot is given, or when pob_anchor="section_corner".
        :param address: Street address printed in the document (e.g. "6361 S Power Rd") —
            alternative way to find the parcel when the APN is missing or illegible.
        :param subdivision: Subdivision/plat name (e.g. "Sundance Groves") — used with `lot`
            as a third way to find the parcel.
        :param lot: Lot number within the subdivision (e.g. "104").
        :param county: County whose GIS to query. Currently supported: maricopa.
        :param pob_anchor: How the Point of Beginning is located (required choice):
            - "parcel_edge": POB lies ON a parcel edge, pob_from_corner_ft from one end.
              Uses pob_edge / pob_from_end / pob_from_corner_ft. (e.g. "a point on the
              north line of the parcel, 122.93 feet west of the NE corner")
            - "parcel_corner": POB IS a named corner of the parcel/lot. Uses pob_corner.
              (e.g. "BEGINNING at the southwest corner of said Lot 104")
            - "section_corner": POB is reached by walking tie course(s) from a named
              corner of a PLSS section. Uses pob_corner, section, township, range_,
              tie_courses_json. (e.g. "COMMENCING at the SE corner of Section 32...
              THENCE N89-29-34W 65.00 FEET; THENCE ... TO THE POINT OF BEGINNING")
        :param pob_edge: (parcel_edge) Which parcel edge: north, south, east, west.
        :param pob_from_end: (parcel_edge) Which end of that edge the tie is measured from.
        :param pob_from_corner_ft: (parcel_edge) Distance in feet from that end to the POB.
        :param pob_corner: (parcel_corner / section_corner) Named corner: NW, NE, SW or SE.
        :param section: (section_corner) PLSS section number as printed, e.g. "31".
        :param township: (section_corner) Township, e.g. "1S" or "T1S".
        :param range_: (section_corner) Range, e.g. "7E" or "R7E".
        :param tie_courses_json: (section_corner) JSON array (same row format as
            line_table_json) of the course(s) walked from the section corner to the POB,
            in order. Use [] or omit if the POB is the section corner itself.

        :returns: If successful, returns a dictionary containing:
            - summary (dict): flat result fields — report these values verbatim.
            - geojson_io_url (str): ready-made link opening the exact polygons on an
              interactive map. Present as a link; NEVER re-type coordinates.
            - geojson (dict): full-precision FeatureCollection for programmatic
              consumers only — do not reproduce it in a chat response.
                 If unsuccessful, returns a ToolError with details about the failure.
        """
        method_name = "map_easement_to_parcel"
        try:
            # ---------------------------------------------- validate inputs
            adapter = COUNTY_GIS.get(county.strip().lower())
            if adapter is None:
                return ToolError(
                    message=f"Unsupported county: {county!r}",
                    suggestions=[
                        f"Supported counties: {', '.join(sorted(COUNTY_GIS))}",
                        "Additional counties require a GIS adapter entry",
                    ],
                )

            anchor = (pob_anchor or "parcel_edge").strip().lower()
            if anchor not in ("parcel_edge", "parcel_corner", "section_corner"):
                return ToolError(
                    message=f"Invalid pob_anchor: {pob_anchor!r}",
                    suggestions=["Use parcel_edge, parcel_corner or section_corner"],
                )

            try:
                lines = _parse_course_rows(line_table_json, "L")
            except Exception as e:
                return ToolError(
                    message=f"Invalid line_table_json: {e}",
                    suggestions=[
                        'Expected: [{"line": "L1", "bearing": "S00-07-32E", "distance_ft": 10.30}, ...]'
                    ],
                )

            # ------------------------------------- 1. traverse + closure check
            verts_local = [(0.0, 0.0)]
            for _, bearing, dist in lines:
                dx, dy = _parse_bearing(bearing, dist)
                x, y = verts_local[-1]
                verts_local.append((x + dx, y + dy))

            cx, cy = verts_local[-1]
            closure_ft = math.hypot(cx, cy)
            perimeter = sum(d for _, _, d in lines)
            area_sqft = 0.5 * abs(
                sum(
                    verts_local[i][0] * verts_local[i + 1][1]
                    - verts_local[i + 1][0] * verts_local[i][1]
                    for i in range(len(verts_local) - 1)
                )
            )
            if closure_ft > perimeter * 0.01:
                return ToolError(
                    message=(
                        f"Traverse does not close: gap of {closure_ft:.2f} ft over a "
                        f"{perimeter:.2f} ft perimeter. The course list is likely "
                        "incomplete, out of order, or mis-read. (Note: strip easements "
                        "described by a centerline plus a width are not closed polygons "
                        "and are not supported yet.)"
                    ),
                    suggestions=[
                        "Verify every course of the traverse is present and in order",
                        "Check bearings/distances were paired correctly",
                    ],
                )

            # ------------------------------------ 2. parcel polygon (if locatable)
            parcel_ring = None
            owner = None
            parcel_found_by = None

            def esc(v: str) -> str:
                return v.replace("'", "''")

            where = None
            if apn.strip():
                where, parcel_found_by = f"{adapter['apn_field']}='{esc(apn.strip())}'", "apn"
            elif address.strip():
                where = (
                    f"UPPER({adapter['address_field']}) LIKE "
                    f"UPPER('%{esc(address.strip())}%')"
                )
                parcel_found_by = "address"
            elif subdivision.strip() and lot.strip():
                where = (
                    f"UPPER({adapter['subdivision_field']}) LIKE "
                    f"UPPER('%{esc(subdivision.strip())}%') "
                    f"AND {adapter['lot_field']}='{esc(lot.strip())}'"
                )
                parcel_found_by = "subdivision+lot"

            if where:
                url = (
                    f"{adapter['parcel_url']}?where={quote(where)}"
                    f"&outFields={adapter['apn_field']},{adapter['owner_field']}"
                    "&returnGeometry=true&outSR=4326&f=json"
                )
                logger.info("Querying %s parcel GIS by %s", county, parcel_found_by)
                gis = await _gis_query(url)
                features = gis.get("features") or []
                if features:
                    parcel_ring = features[0]["geometry"]["rings"][0]
                    attrs = features[0]["attributes"]
                    owner = attrs.get(adapter["owner_field"])
                    apn = attrs.get(adapter["apn_field"]) or apn
                elif anchor != "section_corner":
                    return ToolError(
                        message=(
                            f"No parcel found in {county} county GIS "
                            f"(searched by {parcel_found_by})"
                        ),
                        suggestions=[
                            "Check the APN format (Maricopa uses dashes: 200-18-001S)",
                            "Try the street address or subdivision+lot printed in the document",
                        ],
                    )

            if parcel_ring is None and anchor in ("parcel_edge", "parcel_corner"):
                return ToolError(
                    message=(
                        f"pob_anchor={anchor!r} requires a locatable parcel — provide "
                        "apn, address, or subdivision+lot"
                    ),
                    suggestions=[
                        "Look for an APN, street address, or subdivision/lot in the document",
                        "If the POB is tied to a section corner, use pob_anchor=section_corner",
                    ],
                )

            # --------------------------------------------- 3. locate the POB
            if anchor == "parcel_edge":
                edges = _edge_metrics(parcel_ring)
                max_len = max(e["length_m"] for e in edges)
                candidates = [e for e in edges if e["length_m"] >= max_len * 0.3]
                EDGE_PICK = {
                    "north": ("mid_lat", True),
                    "south": ("mid_lat", False),
                    "east": ("mid_lon", True),
                    "west": ("mid_lon", False),
                }
                stated_edge = pob_edge.strip().lower()
                stated_end = pob_from_end.strip().lower()
                if stated_edge not in EDGE_PICK:
                    return ToolError(
                        message=f"Invalid pob_edge: {pob_edge!r} (use north/south/east/west)"
                    )
                if stated_end not in ("east", "west", "north", "south"):
                    return ToolError(
                        message=f"Invalid pob_from_end: {pob_from_end!r} (use north/south/east/west)"
                    )

                def _place(edge_name, end_name):
                    key, reverse = EDGE_PICK[edge_name]
                    edge = sorted(candidates, key=lambda e: e[key], reverse=reverse)[0]
                    p1, p2 = edge["p1"], edge["p2"]
                    if end_name in ("east", "west"):
                        p_far, p_near = (
                            (p1, p2) if (p1[0] < p2[0]) == (end_name == "east") else (p2, p1)
                        )
                    else:
                        p_far, p_near = (
                            (p1, p2) if (p1[1] < p2[1]) == (end_name == "north") else (p2, p1)
                        )
                    length_ft = edge["length_m"] * FT_PER_M
                    if pob_from_corner_ft > length_ft:
                        return None
                    f = pob_from_corner_ft / length_ft
                    return (
                        p_near[0] + (p_far[0] - p_near[0]) * f,
                        p_near[1] + (p_far[1] - p_near[1]) * f,
                        length_ft,
                    )

                # A granted easement lies within the grantor's parcel. Anchors read
                # off plat drawings are often ambiguous (which edge, which end), so
                # score every hypothesis by containment and auto-correct a stated
                # placement that would put the easement outside the parcel.
                hypotheses = []
                for en in EDGE_PICK:
                    ends = ("east", "west") if en in ("north", "south") else ("north", "south")
                    for endn in ends:
                        placed = _place(en, endn)
                        if placed:
                            score = _containment_score(
                                verts_local, placed[0], placed[1], parcel_ring
                            )
                            # tie-breaks: keep the stated end, then the stated edge
                            hypotheses.append(
                                (
                                    -score,
                                    0 if endn == stated_end else 1,
                                    0 if en == stated_edge else 1,
                                    en,
                                    endn,
                                    placed,
                                )
                            )
                if not hypotheses:
                    return ToolError(
                        message=(
                            f"pob_from_corner_ft ({pob_from_corner_ft}) exceeds every "
                            "candidate parcel edge length"
                        ),
                        suggestions=["Verify the tie dimension"],
                    )
                hypotheses.sort()
                best_score = -hypotheses[0][0]
                stated_placed = _place(stated_edge, stated_end)
                stated_score = (
                    _containment_score(
                        verts_local, stated_placed[0], stated_placed[1], parcel_ring
                    )
                    if stated_placed
                    else -1.0
                )
                correction = ""
                if stated_placed and stated_score >= best_score - 0.15:
                    use_edge, use_end = stated_edge, stated_end
                    pob_lon, pob_lat, edge_len_ft = stated_placed
                else:
                    _, _, _, use_edge, use_end, placed = hypotheses[0]
                    pob_lon, pob_lat, edge_len_ft = placed
                    if (use_edge, use_end) != (stated_edge, stated_end):
                        correction = (
                            f" [auto-corrected from {stated_edge}/{stated_end}: that "
                            f"placement left the easement outside the parcel "
                            f"(containment {max(stated_score, 0):.0%} vs {best_score:.0%})]"
                        )
                        logger.info(
                            "parcel_edge anchor auto-corrected %s/%s -> %s/%s",
                            stated_edge,
                            stated_end,
                            use_edge,
                            use_end,
                        )
                anchor_detail = (
                    f"{pob_from_corner_ft} ft from the {use_end} end of the parcel's "
                    f"{use_edge} edge ({edge_len_ft:.2f} ft long)" + correction
                )

            elif anchor == "parcel_corner":
                pt = _pick_corner(parcel_ring, pob_corner)
                if pt is None:
                    return ToolError(
                        message=f"Invalid pob_corner: {pob_corner!r} (use NW/NE/SW/SE)"
                    )
                pob_lon, pob_lat = pt
                anchor_detail = f"the {pob_corner.upper()} corner of parcel {apn}"

            else:  # section_corner
                if not (section.strip() and township.strip() and range_.strip() and pob_corner.strip()):
                    return ToolError(
                        message=(
                            "section_corner anchoring requires section, township, range_ "
                            "and pob_corner"
                        ),
                        suggestions=[
                            'Example: section="31", township="1S", range_="7E", pob_corner="NW"'
                        ],
                    )
                twp = township.strip().upper().lstrip("T")
                rng = range_.strip().upper().lstrip("R")
                sec = str(int(re.sub(r"\D", "", section)))
                where_trs = (
                    f"Township='T{twp}' AND Range='R{rng}' AND Section='{sec}' "
                    "AND QuarterSection=''"
                )
                url = (
                    f"{adapter['trs_url']}?where={quote(where_trs)}"
                    "&outFields=Township,Range,Section&returnGeometry=true&outSR=4326&f=json"
                )
                logger.info("Querying %s PLSS grid for S%s T%s R%s", county, sec, twp, rng)
                gis = await _gis_query(url)
                features = gis.get("features") or []
                if not features:
                    return ToolError(
                        message=(
                            f"Section {sec} T{twp} R{rng} not found in the {county} "
                            "PLSS grid"
                        ),
                        suggestions=[
                            "Verify section/township/range as printed in the document",
                            "The section may lie outside this county",
                        ],
                    )
                section_ring = features[0]["geometry"]["rings"][0]
                pt = _pick_corner(section_ring, pob_corner)
                if pt is None:
                    return ToolError(
                        message=f"Invalid pob_corner: {pob_corner!r} (use NW/NE/SW/SE)"
                    )
                corner_lon, corner_lat = pt

                # Walk the tie course(s) from the section corner to the POB
                tie_x = tie_y = 0.0
                tie_rows: List[Tuple[str, str, float]] = []
                if tie_courses_json.strip() and tie_courses_json.strip() != "[]":
                    try:
                        tie_rows = _parse_course_rows(tie_courses_json, "T")
                    except Exception as e:
                        return ToolError(
                            message=f"Invalid tie_courses_json: {e}",
                            suggestions=[
                                'Same row format as line_table_json: [{"bearing": "S00-40-01E", "distance_ft": 714.49}, ...]'
                            ],
                        )
                    for _, bearing, dist in tie_rows:
                        dx, dy = _parse_bearing(bearing, dist)
                        tie_x += dx
                        tie_y += dy

                m_lat0, m_lon0 = _meters_per_degree(corner_lat)
                pob_lon = corner_lon + (tie_x / FT_PER_M) / m_lon0
                pob_lat = corner_lat + (tie_y / FT_PER_M) / m_lat0
                tie_len = math.hypot(tie_x, tie_y)
                anchor_detail = (
                    f"{len(tie_rows)} tie course(s), {tie_len:.2f} ft net, from the "
                    f"{pob_corner.upper()} corner of Section {sec} T{twp} R{rng}"
                )

            # -------------------------------------- 4. place the easement ring
            m_lat, m_lon = _meters_per_degree(pob_lat)
            easement_ring = []
            for x_ft, y_ft in verts_local:
                lon = pob_lon + (x_ft / FT_PER_M) / m_lon
                lat = pob_lat + (y_ft / FT_PER_M) / m_lat
                easement_ring.append([round(lon, 8), round(lat, 8)])
            easement_ring[-1] = easement_ring[0]

            # ------------------------------------------------------- 5. GeoJSON
            features_out = []
            if parcel_ring:
                features_out.append(
                    {
                        "type": "Feature",
                        "properties": {
                            "name": f"Parcel APN {apn}",
                            "owner": owner,
                            "source": f"{county} county GIS ({parcel_found_by})",
                            "stroke": "#2166ac",
                            "fill": "#2166ac",
                            "fill-opacity": 0.08,
                        },
                        "geometry": {"type": "Polygon", "coordinates": [parcel_ring]},
                    }
                )
            features_out.append(
                {
                    "type": "Feature",
                    "properties": {
                        "name": "Easement (computed from survey courses)",
                        "area_sqft": round(area_sqft, 1),
                        "closure_ft": round(closure_ft, 3),
                        "stroke": "#b2182b",
                        "fill": "#b2182b",
                        "fill-opacity": 0.45,
                    },
                    "geometry": {"type": "Polygon", "coordinates": [easement_ring]},
                }
            )
            geojson = {"type": "FeatureCollection", "features": features_out}

            # gzip+base64url link — payload is only [A-Za-z0-9-_], immune to
            # chat-UI link mangling, and ~4x smaller than percent-encoded JSON.
            def _dp(pts, eps):
                if len(pts) < 3:
                    return pts
                ax, ay = pts[0]
                bx, by = pts[-1]
                dx, dy = bx - ax, by - ay
                dmax, idx = 0.0, 0
                for i in range(1, len(pts) - 1):
                    px, py = pts[i]
                    if dx == 0 and dy == 0:
                        dist = math.hypot(px - ax, py - ay)
                    else:
                        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
                        dist = math.hypot(px - (ax + t * dx), py - (ay + t * dy))
                    if dist > dmax:
                        dmax, idx = dist, i
                if dmax <= eps:
                    return [pts[0], pts[-1]]
                left = _dp(pts[: idx + 1], eps)
                right = _dp(pts[idx:], eps)
                return left[:-1] + right

            def _simplify_ring(r, eps=2e-6):  # ~0.2 m
                closed = r if r[0] == r[-1] else list(r) + [r[0]]
                out = _dp(closed, eps)
                return out if len(out) >= 4 else closed

            def _round_ring(r, nd=5):
                out = []
                for lon, lat in r:
                    p = [round(lon, nd), round(lat, nd)]
                    if not out or out[-1] != p:
                        out.append(p)
                if out[-1] != out[0]:
                    out.append(out[0])  # GeoJSON rings must be closed
                return out

            compact_features = []
            if parcel_ring:
                compact_features.append(
                    {
                        "type": "Feature",
                        "properties": {"fill-opacity": 0.05},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [_round_ring(_simplify_ring(parcel_ring))],
                        },
                    }
                )
            compact_features.append(
                {
                    "type": "Feature",
                    "properties": {"fill": "#b2182b", "fill-opacity": 0.5},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [_round_ring(easement_ring)],
                    },
                }
            )
            compact = {"type": "FeatureCollection", "features": compact_features}
            raw = json.dumps(compact, separators=(",", ":")).encode()
            packed = base64.urlsafe_b64encode(gzip.compress(raw, mtime=0)).decode().rstrip("=")
            geojson_io_url = "https://geojson.io/?data=gz:" + packed

            precision = f"1:{perimeter / closure_ft:,.0f}" if closure_ft > 0 else "exact"
            logger.info(
                "Easement mapped via %s: closure %s ft (%s), area %.1f sqft, POB %.7f,%.7f",
                anchor,
                round(closure_ft, 3),
                precision,
                area_sqft,
                pob_lat,
                pob_lon,
            )
            return {
                # Flat, small, safe for an LLM to relay verbatim in chat.
                "summary": {
                    "apn": apn or None,
                    "parcel_owner": owner,
                    "parcel_found_by": parcel_found_by,
                    "county": county,
                    "pob_anchor": anchor,
                    "anchor_detail": anchor_detail,
                    "closure_ft": round(closure_ft, 3),
                    "closure_precision": precision,
                    "easement_area_sqft": round(area_sqft, 1),
                    "pob_lat": round(pob_lat, 7),
                    "pob_lon": round(pob_lon, 7),
                },
                # Single opaque link — click to view the exact polygons on a map.
                "geojson_io_url": geojson_io_url,
                # Full-precision payload for programmatic consumers (front-ends
                # should read this from the tool-call event, not from chat text).
                "geojson": geojson,
            }

        except Exception as e:
            logger.error("%s failed: %s", method_name, str(e))
            logger.error(traceback.format_exc())
            return ToolError(
                message=f"{method_name} failed: {str(e)}. Trace available in server logs."
            )
