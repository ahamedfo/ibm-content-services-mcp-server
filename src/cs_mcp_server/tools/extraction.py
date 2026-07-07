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

"""Unified survey-data extraction: ONE vision model reads any easement
document (text PDF or scan) and emits the structured survey data that
map_easement_to_parcel consumes. Replaces per-format text parsing."""

import base64
import json
import logging
import math
import os
import re
import traceback
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import aiohttp
from fastmcp import FastMCP
from mcp.types import ToolAnnotations

from cs_mcp_server.client.graphql_client import (
    GraphQLClient,
    graphql_client_execute_async_wrapper,
)
from cs_mcp_server.tools.easement import _parse_bearing
from cs_mcp_server.utils.common import ToolError

logger = logging.getLogger(__name__)

# Vision backend — any OpenAI-compatible chat-completions endpoint.
# Defaults target Groq's hosted open-source Llama-4 Scout (bake-off winner);
# point at watsonx.ai later by overriding these in the toolkit connection.
VISION_API_URL = os.environ.get(
    "VISION_API_URL", "https://api.groq.com/openai/v1/chat/completions"
)
VISION_MODEL = os.environ.get(
    "VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"
)
MAX_PAGES = 5  # provider image-count limit per request
RENDER_DPI = 200

PROMPT = """You are reading a land-survey easement document (deed pages and/or a plat
drawing). Extract EXACTLY what is printed — do not infer, estimate, or fill gaps.

Return ONLY a JSON object (no prose, no markdown fences):
{
  "line_table": [{"line": "L1", "bearing": "S00-07-32E", "distance_ft": 10.30}, ...],
  "apn": "200-18-001S" or null,
  "address": "6361 S Power Rd" or null,
  "subdivision": "Sundance Groves" or null,
  "lot": "104" or null,
  "county": "maricopa" or null,
  "section_township_range": {"section": 31, "township": "1S", "range": "7E"} or null,
  "pob_anchor": "parcel_edge" | "parcel_corner" | "section_corner" | "unknown",
  "pob_corner": "NW" | "NE" | "SW" | "SE" or null,
  "pob_edge": "north" | "south" | "east" | "west" or null,
  "pob_from_end": "north" | "south" | "east" | "west" or null,
  "pob_from_corner_ft": 122.93 or null,
  "tie_courses": [{"bearing": "S00-40-01E", "distance_ft": 714.49}, ...],
  "legibility": "clean" | "degraded" | "unreadable",
  "notes": "anything ambiguous, illegible, or unusual"
}

Rules:
- bearings as N/S + DD-MM-SS + E/W (convert degree/minute/second symbols to dashes)
- line_table: the courses of the EASEMENT boundary itself, in printed order. If the
  traverse is written as prose ("THENCE North 24 degrees ... 130.00 feet"), convert each
  course to a row in document order. [] if no traverse is present.
- CAREFUL: documents often ALSO describe the parent PROPERTY/parcel boundary (e.g. an
  "Exhibit A" legal description of the grantor's land, or boundary dimensions around a
  plat drawing). Those are NOT the easement. The easement traverse is the one labelled
  L1/L2/... in a LINE TABLE, or the description of the easement/abandonment area itself.
  When both exist, extract ONLY the easement's courses.
- pob_anchor — how the Point of Beginning is located:
  * "parcel_corner": POB IS a named corner of a lot/parcel
    ("BEGINNING at the southwest corner of said Lot 104") -> set pob_corner
  * "section_corner": POB is reached from a named corner of a PLSS Section
    ("COMMENCING at the SE corner of Section 32 ... THENCE ... TO THE POINT OF
    BEGINNING") -> set pob_corner, section_township_range, and put the course(s)
    walked from that corner to the POB in tie_courses (in order)
  * "parcel_edge": POB lies on a parcel boundary line at a stated distance from a
    corner or line end -> set pob_edge, pob_from_end, pob_from_corner_ft
  * "unknown" if you cannot tell — do NOT guess
- tie_courses are the COMMENCING->POB walk only; never duplicate line_table rows there
- PLAT DRAWINGS: the Point of Beginning is where course L1 STARTS. Look for a dimension
  label from a property line end or corner to that exact starting point (e.g. "122.93'").
  If the easement starts ON a property boundary at such a labelled distance:
  pob_anchor="parcel_edge", pob_edge=<which boundary: north/south/east/west>,
  pob_from_end=<which end the dimension is measured from>, pob_from_corner_ft=<distance>.
  Corner-to-corner or boundary-length dimensions of the PROPERTY itself are NOT the POB
  tie — do not put them in tie_courses.
- apn/address/subdivision/lot: only if printed; do NOT guess illegible characters
- county: infer only from explicit text like "Maricopa County" / "Pima County"
"""


def _closure(rows: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    x = y = per = 0.0
    for row in rows:
        try:
            dx, dy = _parse_bearing(str(row["bearing"]), float(row.get("distance_ft", 0)))
        except Exception:
            return None
        x += dx
        y += dy
        per += float(row.get("distance_ft", 0))
    return {"gap_ft": math.hypot(x, y), "perimeter_ft": per}


def register_extraction_tools(mcp: FastMCP, graphql_client: GraphQLClient) -> None:
    @mcp.tool(
        name="extract_survey_data",
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def extract_survey_data(
        identifier: str,
    ) -> Union[Dict[str, Any], ToolError]:
        """
        Reads an easement document with a vision model and returns the structured
        survey data needed to map it — regardless of whether the document is a
        clean text PDF, an OCR-quality scan, or a pure image scan. The document's
        pages are rendered to images in memory and read by the model; nothing is
        written to disk and no lock is placed on the document.

        Call this INSTEAD of parsing document text yourself. Pass its outputs
        directly to map_easement_to_parcel (the *_json fields are pre-formatted
        for that tool's parameters).

        :param identifier: The document id or path (required). GUID or repository path.

        :returns: If successful, returns a dictionary containing:
            - line_table (list) and line_table_json (str, ready for map_easement_to_parcel)
            - closure_ok (bool), closure_gap_ft, closure_perimeter_ft — arithmetic check
              of the extracted courses; if false, treat extraction as unreliable
            - apn, address, subdivision, lot, county — parcel identification (null if absent)
            - pob_anchor, pob_corner, pob_edge, pob_from_end, pob_from_corner_ft,
              section, township, range_, tie_courses_json — anchor fields, matching
              map_easement_to_parcel's parameters
            - page_count, pages_read, legibility, notes
                 If unsuccessful, returns a ToolError with details about the failure.
        """
        method_name = "extract_survey_data"
        try:
            api_key = os.environ.get("VISION_API_KEY") or os.environ.get("GROQ_API_KEY")
            if not api_key:
                return ToolError(
                    message="Vision backend not configured (VISION_API_KEY missing)",
                    suggestions=[
                        "Add VISION_API_KEY (and optionally VISION_API_URL/VISION_MODEL) "
                        "to the toolkit connection",
                        "Fallback: use get_document_pdf_text and parse the text manually",
                    ],
                )

            from cs_mcp_server.tools.documents import _normalize_identifier

            identifier = _normalize_identifier(identifier)

            # ---------------- 1. fetch document bytes from the repository
            query = """
            query ($object_store_name: String!, $identifier: String!) {
                document(repositoryIdentifier: $object_store_name, identifier: $identifier) {
                    id
                    className
                    currentVersion{
                        contentElements{
                            ... on ContentTransferType {
                                retrievalName
                                contentType
                                contentSize
                                downloadUrl
                            }
                        }
                    }
                }
            }
            """
            variables = {
                "object_store_name": graphql_client.object_store,
                "identifier": identifier,
            }
            response = await graphql_client_execute_async_wrapper(
                logger, method_name, graphql_client, query=query, variables=variables
            )
            if isinstance(response, ToolError):
                return response
            if not response.get("data") or not response["data"].get("document"):
                return ToolError(
                    message=f"Document not found with identifier: {identifier}",
                    suggestions=["Check the document ID or path"],
                )
            document = response["data"]["document"]
            elements = [
                e
                for e in (document.get("currentVersion") or {}).get("contentElements") or []
                if e.get("downloadUrl")
            ]
            if not elements:
                return ToolError(
                    message=f"Document has no downloadable content: {identifier}"
                )
            content = await graphql_client.download_content_bytes_async(
                download_url=elements[0]["downloadUrl"]
            )
            content_type = (elements[0].get("contentType") or "").lower()

            # ---------------- 2. render pages to images (in memory)
            import fitz  # pymupdf

            doc = fitz.open(stream=content, filetype=None if content_type else "pdf")
            page_count = len(doc)

            # Prefer pages whose text layer mentions survey keywords; for pure
            # scans, exhibits usually live at the END of recorded documents.
            KEYWORDS = ("THENCE", "LINE TABLE", "EXHIBIT", "APN", "COMMENCING", "BEGINNING")
            scored = []
            for i in range(page_count):
                try:
                    text = (doc[i].get_text() or "").upper()
                except Exception:
                    text = ""
                score = sum(k in text for k in KEYWORDS)
                scored.append((score, i))
            has_text = any(s for s, _ in scored)
            if has_text:
                picked = [i for s, i in sorted(scored, key=lambda t: (-t[0], t[1])) if s > 0][:MAX_PAGES]
                if not picked:
                    picked = list(range(min(page_count, MAX_PAGES)))
                picked = sorted(picked)
            elif page_count <= MAX_PAGES:
                picked = list(range(page_count))
            else:
                # first page (identification) + last pages (exhibits)
                picked = [0] + list(range(page_count - (MAX_PAGES - 1), page_count))

            images_b64 = []
            for i in picked:
                pix = doc[i].get_pixmap(dpi=RENDER_DPI)
                images_b64.append(base64.b64encode(pix.tobytes("png")).decode())
            doc.close()

            # ---------------- 3. vision call (retry once if traverse won't close)
            async def call_vision() -> Dict[str, Any]:
                content_parts: List[Dict[str, Any]] = [{"type": "text", "text": PROMPT}]
                for b64 in images_b64:
                    content_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        }
                    )
                payload = {
                    "model": VISION_MODEL,
                    "messages": [{"role": "user", "content": content_parts}],
                    "temperature": 0,
                    "max_tokens": 4000,
                }
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        VISION_API_URL,
                        headers={"Authorization": f"Bearer {api_key}"},
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=180),
                    ) as resp:
                        body = await resp.text()
                        if resp.status != 200:
                            raise RuntimeError(
                                f"Vision API HTTP {resp.status}: {body[:200]}"
                            )
                text = json.loads(body)["choices"][0]["message"]["content"]
                start, end = text.find("{"), text.rfind("}")
                return json.loads(text[start : end + 1])

            import asyncio as _asyncio

            def _closes(chk):
                return bool(chk and chk["gap_ft"] <= max(chk["perimeter_ft"] * 0.01, 0.5))

            out = await call_vision()
            rows = out.get("line_table") or []
            check = _closure(rows) if rows else None
            closure_ok = _closes(check)
            if rows and not closure_ok:
                # One retry after the provider's per-minute token window resets.
                # If the retry itself fails (e.g. rate limit), keep pass-1 results
                # with closure_ok=False rather than losing everything.
                logger.info("Traverse from first pass does not close; retrying once")
                try:
                    await _asyncio.sleep(65)
                    out2 = await call_vision()
                    rows2 = out2.get("line_table") or []
                    check2 = _closure(rows2) if rows2 else None
                    if _closes(check2):
                        out, rows, check, closure_ok = out2, rows2, check2, True
                except Exception as retry_err:
                    logger.warning("Extraction retry failed: %s", str(retry_err)[:150])

            # ---------------- 4. flatten into map_easement_to_parcel-ready fields
            str_info = out.get("section_township_range") or {}
            tie_rows = out.get("tie_courses") or []
            result = {
                "document_id": document["id"],
                "page_count": page_count,
                "pages_read": [i + 1 for i in picked],
                "legibility": out.get("legibility"),
                "notes": out.get("notes"),
                "line_table": rows,
                "line_table_json": json.dumps(rows),
                "closure_ok": closure_ok,
                "closure_gap_ft": round(check["gap_ft"], 3) if check else None,
                "closure_perimeter_ft": round(check["perimeter_ft"], 2) if check else None,
                "apn": out.get("apn"),
                "address": out.get("address"),
                "subdivision": out.get("subdivision"),
                "lot": str(out.get("lot")) if out.get("lot") is not None else None,
                "county": (out.get("county") or "maricopa").lower(),
                "pob_anchor": out.get("pob_anchor") or "unknown",
                "pob_corner": out.get("pob_corner"),
                "pob_edge": out.get("pob_edge"),
                "pob_from_end": out.get("pob_from_end"),
                "pob_from_corner_ft": out.get("pob_from_corner_ft"),
                "section": str(str_info.get("section")) if str_info.get("section") else None,
                "township": str_info.get("township"),
                "range_": str_info.get("range"),
                "tie_courses_json": json.dumps(tie_rows),
            }
            logger.info(
                "Extracted %d courses (closure_ok=%s), anchor=%s, from pages %s of %s",
                len(rows),
                closure_ok,
                result["pob_anchor"],
                result["pages_read"],
                identifier,
            )
            return result

        except Exception as e:
            logger.error("%s failed: %s", method_name, str(e))
            logger.error(traceback.format_exc())
            return ToolError(
                message=f"{method_name} failed: {str(e)}. Trace available in server logs."
            )
