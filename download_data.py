import os
import requests
import mwxml
import mwparserfromhell
import bz2
import re
from tqdm import tqdm


class WikiDumpProcessor:
    def __init__(self, dump_file, output_dir="./wikivoyage_data", root_filter=None):
        """
        :param dump_file: Local filename (e.g., enwikivoyage-latest-pages-articles.xml.bz2)
        :param output_dir: Directory where Markdown files will be saved.
        :param root_filter: (Optional) A keyword (e.g., "China" or "Singapore").
                            Only pages belonging to this region's hierarchy will be saved.
        """
        self.dump_file = dump_file
        self.output_dir = output_dir
        self.root_filter = root_filter
        self.count = 0
        self.base_url = "https://dumps.wikimedia.org/enwikivoyage/latest/"
        self.download_filename = "enwikivoyage-latest-pages-articles.xml.bz2"

        # Dictionary to store parent-child relationships
        # Example: { "Sentosa": "Singapore", "Zhejiang": "East China" }
        self.parent_map = {}

    def _download_dump(self):
        """Automatically downloads the latest Wikivoyage dump file."""
        url = self.base_url + self.download_filename
        print(f"🔍 File {self.dump_file} not found locally. Preparing to download...")
        print(f"⬇️ Download URL: {url}")
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))

            with open(self.dump_file, 'wb') as f, tqdm(
                    desc="Downloading", total=total_size, unit='iB',
                    unit_scale=True, unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(1024):
                    size = f.write(data)
                    bar.update(size)
            print(f"\n✅ Download complete.")
        except Exception as e:
            print(f"\n❌ Download failed: {e}")
            if os.path.exists(self.dump_file): os.remove(self.dump_file)
            raise e

    def _build_hierarchy(self):
        """
        Phase 1: Scans the entire dump to build a hierarchy tree.
        Logic: Uses both 'slash' naming conventions (Parent/Child) and {{IsPartOf}} templates.
        """
        print("Phase 1/2: Building Hierarchy Map...")
        print("    (Scanning file to determine folder structure, please wait...)")

        # Regex to find {{IsPartOf|Parent}} tag (case insensitive)
        pattern = re.compile(r'\{\{ispartof\|([^}]+)\}\}', re.IGNORECASE)

        count = 0
        try:
            with bz2.open(self.dump_file, 'rt', encoding='utf-8', errors='ignore') as f:
                dump = mwxml.Dump.from_file(f)
                for page in dump:
                    if page.namespace != 0 or page.redirect:
                        continue

                    # --- Strategy A: Slash Logic (Implicit Hierarchy) ---
                    # If title is "Singapore/Sentosa", the parent is automatically "Singapore"
                    if '/' in page.title:
                        parent_by_slash = page.title.rsplit('/', 1)[0].strip()
                        self.parent_map[page.title] = parent_by_slash

                    # --- Strategy B: Template Logic (Explicit Hierarchy) ---
                    # Templates are often more accurate for connecting disjointed names
                    # (e.g., connecting "Zhejiang" to "East China")
                    for revision in page:
                        text = revision.text
                        if text:
                            match = pattern.search(text)
                            if match:
                                parent_by_template = match.group(1).strip()
                                # Handle cases like {{IsPartOf|China|Asia}} -> take the first one
                                parent_by_template = parent_by_template.split('|')[0].strip()
                                self.parent_map[page.title] = parent_by_template
                        break

                    count += 1
                    if count % 20000 == 0:
                        print(f"    Scanned {count} pages...")

            print(f"✅ Hierarchy Map built! Recorded {len(self.parent_map)} relationships.")

        except Exception as e:
            print(f"❌ Error building hierarchy: {e}")

    def _get_full_path(self, title):
        """
        Recursively looks up parents to generate the full path list.
        Example: "Sentosa" -> ["Asia", "Southeast Asia", "Singapore", "Sentosa"]
        """
        path = [title]
        current = title
        visited = set()  # Prevent infinite loops

        while current in self.parent_map:
            parent = self.parent_map[current]

            # Cleaning: remove anchors like "China#Get in"
            parent = parent.split('#')[0].strip()

            if parent in visited or parent == current or not parent:
                break
            visited.add(parent)
            path.insert(0, parent)
            current = parent

            # Safety break for excessively deep recursion
            if len(path) > 15:
                break

        return path

    def _convert_wikitext_to_markdown(self, wikitext):
        """Parses Wikitext and converts it to readable Markdown."""
        if not wikitext: return ""
        try:
            wikicode = mwparserfromhell.parse(wikitext)
        except:
            return wikitext

        lines = []
        for node in wikicode.nodes:
            # 1. Headings (== Title == becomes ## Title)
            if isinstance(node, mwparserfromhell.nodes.Heading):
                level = node.level
                title = node.title.strip_code().strip()
                lines.append(f"\n{'#' * level} {title}\n")

            # 2. Templates (See/Do/Eat Listings)
            elif isinstance(node, mwparserfromhell.nodes.Template):
                name = node.name.strip().lower()
                if name in ['see', 'do', 'buy', 'eat', 'drink', 'sleep', 'listing']:
                    # Use listing name as sub-heading
                    if node.has("name"):
                        item_name = node.get("name").value.strip_code().strip()
                        if item_name:
                            lines.append(f"\n#### {item_name}\n")

                    # Extract description content
                    content_parts = []
                    for param in ['content', 'description', 'alt', 'address', 'directions']:
                        if node.has(param):
                            val = node.get(param).value.strip_code().strip()
                            if val: content_parts.append(val)
                    if content_parts:
                        lines.append(" ".join(content_parts) + "\n")

            # 3. Plain Text
            elif isinstance(node, mwparserfromhell.nodes.Text):
                text = str(node).strip()
                if text:
                    clean_text = text.replace("'''", "").replace("''", "")
                    if len(clean_text) > 2:
                        lines.append(clean_text)

            # 4. HTML Tags (<ul>, <li>)
            elif isinstance(node, mwparserfromhell.nodes.Tag):
                try:
                    lines.append(node.strip_code())
                except:
                    pass

        return "\n".join(lines)

    def _save_page(self, title, wikitext):
        # 1. Get full hierarchical path
        path_list = self._get_full_path(title)

        # 2. Apply Root Filter
        if self.root_filter:
            # Check if the filter keyword exists anywhere in the path
            in_path = any(self.root_filter.lower() in p.lower() for p in path_list)
            if not in_path:
                return

        clean_text = self._convert_wikitext_to_markdown(wikitext)
        if not clean_text or len(clean_text) < 50: return

        # 3. Path and Filename Cleaning
        # Sanitize strings for file system usage
        safe_path_list = [p.replace(':', '_').replace('?', '').strip() for p in path_list]

        # Directory: Join all elements except the last one
        if len(safe_path_list) > 1:
            relative_dir = os.path.join(*safe_path_list[:-1])
        else:
            relative_dir = ""

        # Filename: The last element in the path
        raw_filename = safe_path_list[-1]

        # [Fix]: If the page name is "Singapore/Sentosa", we only want "Sentosa.md"
        # because "Singapore" is already the parent folder.
        if '/' in raw_filename:
            final_filename = raw_filename.rsplit('/', 1)[-1]
        else:
            final_filename = raw_filename

        file_name = f"{final_filename}.md"
        dir_path = os.path.join(self.output_dir, relative_dir)

        try:
            os.makedirs(dir_path, exist_ok=True)
            full_path = os.path.join(dir_path, file_name)

            with open(full_path, 'w', encoding='utf-8') as f:
                # Add Breadcrumb Metadata
                breadcrumb = " > ".join(path_list)
                f.write(f"# {title}\n")
                f.write(f"> Hierarchy: {breadcrumb}\n\n")
                f.write(clean_text)

            self.count += 1
            if self.count % 1000 == 0:
                print(f"Processed {self.count} pages | Latest: {file_name}")

        except OSError:
            # Ignore errors related to paths being too long (common in Windows)
            pass

    def process(self):
        # Step 1: Check Download
        if not os.path.exists(self.dump_file):
            self._download_dump()
        else:
            print(f"✅ Found local file: {self.dump_file}")

        # Step 2: Build Hierarchy (Phase 1)
        self._build_hierarchy()

        # Step 3: Extract Content (Phase 2)
        print(f"Phase 2/2: Generating Markdown files...")
        try:
            with bz2.open(self.dump_file, 'rt', encoding='utf-8', errors='ignore') as f:
                dump = mwxml.Dump.from_file(f)
                for page in dump:
                    if page.namespace != 0 or page.redirect: continue
                    for revision in page:
                        self._save_page(page.title, revision.text)
                        break  # Only process the latest revision

            print(f"\n🎉 Task Complete! Extracted {self.count} documents to {self.output_dir}")
        except Exception as e:
            print(f"❌ Process interrupted: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # Configuration
    DUMP_FILE = "enwikivoyage-latest-pages-articles.xml.bz2"

    # Example 1: Extract "Singapore" data
    # This will now correctly structure "Singapore/Sentosa" into folders
    # processor = WikiDumpProcessor(
    #     dump_file=DUMP_FILE,
    #     output_dir="./wikivoyage_sg",
    #     root_filter="Singapore"
    # )

    # Example 2: Extract "China" data (Uncomment to use)
    processor = WikiDumpProcessor(
        dump_file=DUMP_FILE,
        output_dir="./wikivoyage_global"
    )

    processor.process()