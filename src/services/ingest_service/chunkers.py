from pathlib import Path
import json

def _as_int_year(value) -> int | None:
    try:
        if value is None:
            return None
        s = str(value).strip()
        return int(s) if s else None
    except Exception:
        return None


def _split_more_details(text: str, max_chars: int) -> list[str]:
    if not text:
        return []

    t = str(text).strip()
    if not t:
        return []

    if len(t) <= max_chars:
        return [t]

    parts = [p.strip() for p in t.splitlines() if p.strip()]
    if not parts:
        return [t[:max_chars]]

    out: list[str] = []
    buf = ""
    for p in parts:
        candidate = p if not buf else f"{buf}\n{p}"
        if len(candidate) <= max_chars:
            buf = candidate
            continue

        if buf:
            out.append(buf)
            buf = ""

        if len(p) <= max_chars:
            buf = p
        else:
            for i in range(0, len(p), max_chars):
                out.append(p[i : i + max_chars])

    if buf:
        out.append(buf)

    return out


def custom_chunker(
    data_path: Path,
    *,
    split_more_details: bool = True,
    more_details_max_chars: int = 1200,
    include_contact_in_text: bool = False,
) -> list[dict]:
    try:
        try:
            text = Path(data_path).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = Path(data_path).read_text(encoding="utf-8-sig")
        data = json.loads(text)
    except Exception as e:
        print(f"Error loading data from {data_path}: {e}")
        return []
    
    chunks: list[dict] = []
    for item in data:
        source_link = item.get("source_link")
        title = item.get("title")
        price = item.get("price")
        contact = item.get("contact")
        location = item.get("location")
        year_raw = item.get("year")
        make = item.get("make")
        model = item.get("model")
        fuel_type = item.get("fuel_type")
        gear = item.get("gear")
        condition = item.get("condition")
        mileage = item.get("mileage")
        more_details = item.get("more_details") or ""

        metadata = {
            "source_link": source_link,
            "title": title,
            "price": price,
            "contact": contact,
            "location": location,
            "year": _as_int_year(year_raw),
            "make": make,
            "model": model,
            "fuel_type": fuel_type,
            "gear": gear,
            "condition": condition,
            "mileage": mileage,
        }

        base_lines: list[str] = []
        if title:
            base_lines.append(f"Title: {title}")
        if price:
            base_lines.append(f"Price: {price}")
        if location:
            base_lines.append(f"Location: {location}")
        if year_raw:
            base_lines.append(f"Year: {year_raw}")
        if make:
            base_lines.append(f"Make: {make}")
        if model:
            base_lines.append(f"Model: {model}")
        if fuel_type:
            base_lines.append(f"Fuel Type: {fuel_type}")
        if gear:
            base_lines.append(f"Gear: {gear}")
        if condition:
            base_lines.append(f"Condition: {condition}")
        if mileage:
            base_lines.append(f"Mileage: {mileage}")
        if include_contact_in_text and contact:
            base_lines.append(f"Contact: {contact}")
        if source_link:
            base_lines.append(f"Source Link: {source_link}")

        base_text = "\n".join(base_lines).strip()

        if split_more_details:
            md_chunks = _split_more_details(more_details, more_details_max_chars)
            if md_chunks:
                for md in md_chunks:
                    text = f"{base_text}\nMore Details: {md}".strip()
                    chunks.append({"text": text, "metadata": metadata})
            else:
                chunks.append({"text": base_text, "metadata": metadata})
        else:
            text = f"{base_text}\nMore Details: {str(more_details).strip()}".strip()
            chunks.append({"text": text, "metadata": metadata})

    return chunks