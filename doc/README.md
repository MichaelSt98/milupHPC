# API documentation using Doxygen

* Generate documentation from source code: [Doxygen](https://doxygen.nl/)

| File | Description |
| ---- | ----------- |
| Doxyfile | Doxygen file/settings |
| DoxygenFiles/doxygen-custom.css | custom CSS file |
| DoxygenFiles/header.html | custom HTML header |
| DoxygenFiles/footer.html | custom HTML footer |
| Mainpage.md | main page for Doxygen documentation |

## Usage

* the doxygen documentation generation is included into the Makefile: `make doc`, which corresponds to
	* `doxygen ./doc/Doxyfile &> ./doc/doxygen.log`
	* `cp -r "./doc/html/" "./docs/"`
* open in your local browser
	* either `docs/index.html`
	* or `doc/html/index.html`  