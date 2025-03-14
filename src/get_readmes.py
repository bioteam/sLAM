import requests
import base64
import time
import os
import random
from tqdm import tqdm

# GitHub API token - create one at https://github.com/settings/tokens
# No special permissions required for public repos
# Replace with your token or use environment variable
GITHUB_TOKEN = os.environ.get("GITHUB_API_KEY")

# Directory to save README files
OUTPUT_DIR = "github_readmes"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Headers for GitHub API requests
headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}


def get_repositories(per_page=100, num_pages=10):
    """Fetch a list of public repositories"""
    repos = []

    # Different ways to search for repositories to get a diverse set
    search_queries = [
        "stars:>100",  # Popular repos
        "stars:10..100",  # Medium popularity
        "created:>2022-01-01",  # Recent repos
        "language:python",  # Different languages
        "language:javascript",
        "language:java",
        "language:go",
        "language:rust",
        "topic:machine-learning",  # Various topics
        "topic:web-development",
    ]

    for query in search_queries:
        if len(repos) >= 1000:
            break

        print(f"Searching for repositories with query: {query}")
        for page in range(1, num_pages + 1):
            if len(repos) >= 1000:
                break

            url = f"https://api.github.com/search/repositories?q={query}&sort=updated&per_page={per_page}&page={page}"
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                result = response.json()
                repos.extend(repo["full_name"] for repo in result["items"])
                print(
                    f"Found {len(result['items'])} repositories on page {page}"
                )

                # Check if we've reached the last page
                if len(result["items"]) < per_page:
                    break

                # Respect GitHub's rate limits
                time.sleep(2)
            else:
                print(f"Error fetching repositories: {response.status_code}")
                print(response.json())
                time.sleep(10)  # Wait longer on error

    # Return unique repositories, limited to 1000
    return list(set(repos))[:1000]


def get_readme(repo_name):
    """Fetch the README file from a repository"""
    url = f"https://api.github.com/repos/{repo_name}/readme"

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            content = response.json().get("content", "")
            encoding = response.json().get("encoding", "")

            if encoding == "base64":
                return base64.b64decode(content).decode(
                    "utf-8", errors="replace"
                )
            return content
    except Exception as e:
        print(f"Error fetching README for {repo_name}: {e}")
        return None


def save_readme(repo_name, content):
    """Save README content to a file"""
    # Clean the repo name to use as a filename
    safe_name = repo_name.replace("/", "_").replace("\\", "_")
    file_path = os.path.join(OUTPUT_DIR, f"{safe_name}_README.md")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"# Repository: {repo_name}\n\n")
        f.write(content)

    return file_path


def main():
    # Get list of repositories
    print("Fetching repository list...")
    repositories = get_repositories()
    print(f"Found {len(repositories)} unique repositories")

    # Limit to 1000
    repositories = repositories[:1000]

    # Fetch and save README files
    print("Fetching README files...")
    successful_count = 0

    for repo in tqdm(repositories):
        readme_content = get_readme(repo)

        if readme_content:
            save_readme(repo, readme_content)
            successful_count += 1

            # Log progress
            if successful_count % 50 == 0:
                print(f"Saved {successful_count} README files so far")

            # Respect GitHub's rate limits - random delay between 1-3 seconds
            time.sleep(1 + 2 * random.random())

    print(
        f"Successfully saved {successful_count} README files to {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
