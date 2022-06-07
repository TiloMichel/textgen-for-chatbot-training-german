import re


def remove_jsession_token(url: str) -> str:
    test = re.sub(r";jsessionid=[^?]*", "", url)
    return test