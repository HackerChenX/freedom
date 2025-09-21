import gettext
from pathlib import Path
from typing import Union


localedir: Path = Path(__file__).parent

translations: Union[gettext.GNUTranslations, gettext.NullTranslations] = gettext.translation("vnpy", localedir=localedir, fallback=True)

_ = translations.gettext
