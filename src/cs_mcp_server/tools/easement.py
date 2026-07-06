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
document onto real-world coordinates using county parcel GIS data."""

import json
import logging
import math
import re
import traceback
from typing import Any, Dict, List, Union
from urllib.parse import quote

import aiohttp
from fastmcp import FastMCP
from mcp.types import ToolAnnotations

from cs_mcp_server.utils.common import ToolError

logger = logging.getLogger(__name__)

FT_PER_M = 3.28083333

# County GIS adapter registry. Each adapter defines the ArcGIS REST parcel
# layer and how to query an APN against it. Add counties here as needed.
COUNTY_GIS: Dict[str, Dict[str, str]] = {
    "maricopa": {
        "query_url": (
            "https://gis.maricopa.gov/arcgis/rest/services/IndividualService/"
            "Parcel/MapServer/1/query"
        ),
        "apn_field": "APNDash",
        "owner_field": "OwnerName",
    },
}


def _parse_bearing(bearing: str, dist_ft: float):
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


def _meters_per_degree(lat_deg: float):
    phi = math.radians(lat_deg)
    m_lat = 111132.92 - 559.82 * math.cos(2 * phi) + 1.175 * math.cos(4 * phi)
    m_lon = 111412.84 * math.cos(phi) - 93.5 * math.cos(3 * phi)
    return m_lat, m_lon


def _edge_metrics(ring: List[List[float]]):
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


def register_easement_tools(mcp: FastMCP) -> None:
    @mcp.tool(
        name="map_easement_to_parcel",
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def map_easement_to_parcel(
        apn: str,
        line_table_json: str,
        pob_from_corner_ft: float,
        county: str = "maricopa",
        pob_edge: str = "north",
        pob_from_end: str = "east",
    ) -> Union[Dict[str, Any], ToolError]:
        """
        Places an easement survey traverse onto real-world map coordinates and
        returns GeoJSON. The parcel polygon is fetched from the county assessor's
        public GIS by APN; the traverse (from the survey plat's line table) is
        chained, closure-checked, and anchored to the parcel boundary.

        This tool performs no repository operations — it is a pure geospatial
        computation using public county GIS data.

        :param apn: Assessor's Parcel Number as printed on the survey plat (e.g. "200-18-001S").
        :param line_table_json: JSON array of the plat's line table in traverse order, e.g.
            '[{"line": "L1", "bearing": "S00-07-32E", "distance_ft": 10.30}, ...]'.
            Bearings accept degree symbols or dashes (N89°51'34"E or N89-51-34E).
            Note: PDF text extraction often yields the table column-wise — pair the
            Nth line label with the Nth bearing and Nth distance when constructing this.
        :param pob_from_corner_ft: Distance in feet from a parcel corner to the traverse's
            Point of Beginning, measured along the parcel edge given by pob_edge.
            On survey plats this appears as a tie dimension (e.g. "122.93'").
        :param county: County whose GIS to query. Currently supported: maricopa.
        :param pob_edge: Parcel edge the POB lies on: north, south, east or west. Default north.
        :param pob_from_end: Which end of that edge the tie is measured from
            (east/west for north/south edges; north/south for east/west edges). Default east.

        :returns: If successful, returns a dictionary containing:
            - parcel (dict): apn, owner, and the matched edge length in feet.
            - traverse (dict): closure_ft, closure_precision (e.g. "1:44079"), area_sqft.
            - pob (dict): lat/lon of the Point of Beginning.
            - geojson (dict): FeatureCollection with the parcel and easement polygons,
              ready for geojson.io or any GIS system.
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

            try:
                rows = json.loads(line_table_json)
                lines = [
                    (
                        str(row.get("line", f"L{i + 1}")),
                        str(row["bearing"]),
                        float(row.get("distance_ft", row.get("distance"))),
                    )
                    for i, row in enumerate(rows)
                ]
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
                        f"{perimeter:.2f} ft perimeter. The line table is likely "
                        "incomplete, out of order, or mis-parsed."
                    ),
                    suggestions=[
                        "Verify every line of the plat's line table is present and in order",
                        "Check bearings/distances were paired correctly (column-wise extraction)",
                    ],
                )

            # ------------------------------------ 2. parcel polygon from county GIS
            where = quote(f"{adapter['apn_field']}='{apn}'")
            url = (
                f"{adapter['query_url']}?where={where}"
                f"&outFields={adapter['apn_field']},{adapter['owner_field']}"
                "&returnGeometry=true&outSR=4326&f=json"
            )
            logger.info("Querying %s county GIS for APN %s", county, apn)
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status != 200:
                        return ToolError(
                            message=f"County GIS query failed with HTTP {resp.status}"
                        )
                    gis = await resp.json(content_type=None)

            features = gis.get("features") or []
            if not features:
                return ToolError(
                    message=f"No parcel found in {county} county GIS for APN {apn!r}",
                    suggestions=[
                        "Check the APN format (e.g. Maricopa uses dashes: 200-18-001S)",
                        "Verify the parcel is in the selected county",
                    ],
                )
            ring = features[0]["geometry"]["rings"][0]
            owner = features[0]["attributes"].get(adapter["owner_field"])

            # --------------------------------------------- 3. anchor the traverse
            edges = _edge_metrics(ring)
            max_len = max(e["length_m"] for e in edges)
            candidates = [e for e in edges if e["length_m"] >= max_len * 0.3]
            key, reverse = {
                "north": ("mid_lat", True),
                "south": ("mid_lat", False),
                "east": ("mid_lon", True),
                "west": ("mid_lon", False),
            }.get(pob_edge.strip().lower(), (None, None))
            if key is None:
                return ToolError(
                    message=f"Invalid pob_edge: {pob_edge!r} (use north/south/east/west)"
                )
            edge = sorted(candidates, key=lambda e: e[key], reverse=reverse)[0]

            p1, p2 = edge["p1"], edge["p2"]
            # Order the edge ends so the tie is measured from the requested end
            end = pob_from_end.strip().lower()
            if end in ("east", "west"):
                p_far, p_near = (p1, p2) if (p1[0] < p2[0]) == (end == "east") else (p2, p1)
            elif end in ("north", "south"):
                p_far, p_near = (p1, p2) if (p1[1] < p2[1]) == (end == "north") else (p2, p1)
            else:
                return ToolError(
                    message=f"Invalid pob_from_end: {pob_from_end!r} (use north/south/east/west)"
                )

            edge_len_ft = edge["length_m"] * FT_PER_M
            if pob_from_corner_ft > edge_len_ft:
                return ToolError(
                    message=(
                        f"pob_from_corner_ft ({pob_from_corner_ft}) exceeds the parcel's "
                        f"{pob_edge} edge length ({edge_len_ft:.2f} ft)"
                    ),
                    suggestions=["Verify the tie dimension and the pob_edge/pob_from_end choice"],
                )

            frac = pob_from_corner_ft / edge_len_ft
            pob_lon = p_near[0] + (p_far[0] - p_near[0]) * frac
            pob_lat = p_near[1] + (p_far[1] - p_near[1]) * frac
            m_lat, m_lon = _meters_per_degree(pob_lat)

            easement_ring = []
            for x_ft, y_ft in verts_local:
                lon = pob_lon + (x_ft / FT_PER_M) / m_lon
                lat = pob_lat + (y_ft / FT_PER_M) / m_lat
                easement_ring.append([round(lon, 8), round(lat, 8)])
            easement_ring[-1] = easement_ring[0]

            # ------------------------------------------------------- 4. GeoJSON
            geojson = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "name": f"Parcel APN {apn}",
                            "owner": owner,
                            "source": f"{county} county GIS",
                            "stroke": "#2166ac",
                            "fill": "#2166ac",
                            "fill-opacity": 0.08,
                        },
                        "geometry": {"type": "Polygon", "coordinates": [ring]},
                    },
                    {
                        "type": "Feature",
                        "properties": {
                            "name": "Easement (computed from survey line table)",
                            "area_sqft": round(area_sqft, 1),
                            "closure_ft": round(closure_ft, 3),
                            "stroke": "#b2182b",
                            "fill": "#b2182b",
                            "fill-opacity": 0.45,
                        },
                        "geometry": {"type": "Polygon", "coordinates": [easement_ring]},
                    },
                ],
            }

            precision = f"1:{perimeter / closure_ft:,.0f}" if closure_ft > 0 else "exact"
            logger.info(
                "Easement mapped: closure %s ft (%s), area %.1f sqft, POB %.7f,%.7f",
                round(closure_ft, 3),
                precision,
                area_sqft,
                pob_lat,
                pob_lon,
            )
            return {
                "parcel": {
                    "apn": apn,
                    "owner": owner,
                    "matched_edge": pob_edge,
                    "matched_edge_length_ft": round(edge_len_ft, 2),
                },
                "traverse": {
                    "closure_ft": round(closure_ft, 3),
                    "closure_precision": precision,
                    "area_sqft": round(area_sqft, 1),
                },
                "pob": {"lat": round(pob_lat, 8), "lon": round(pob_lon, 8)},
                "geojson": geojson,
            }

        except Exception as e:
            logger.error("%s failed: %s", method_name, str(e))
            logger.error(traceback.format_exc())
            return ToolError(
                message=f"{method_name} failed: {str(e)}. Trace available in server logs."
            )
