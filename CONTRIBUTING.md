# Contribution guidelines

## Adding a new paper

You can add a new paper by making a Pull Request.
Please ensure your pull request adheres to the following guidelines:

- Search included papers before adding a new one, as yours may be a duplicate.
- Please make sure the paper is put in the right order.
- If the paper is both avilable on Arxiv and published in a conference, please prioritize the conference link.

To display the citation of the paper, you need to go to [semanticscholar.org](https://www.semanticscholar.org/), search for the paper, and get the paper ID from the URL. Then you can replace `<PAPER-ID>` in the following code with the paper ID (without the angle brackets).

```html
<br> ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F<PAPER-ID>%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)
```

For example, the paper ID for the following URL is `7e269bfabb451765a16ca0357de6b497cefb60bf`, i.e. the paper ID is the last part of the URL.

```url
https://www.semanticscholar.org/paper/Grammar-Constrained-Decoding-for-Structured-NLP-Geng-Josifosky/7e269bfabb451765a16ca0357de6b497cefb60bf
```

## Updating the information of a paper

It may happen that the information of a paper is not correct. You can update the information by making a Pull Request and explaining the issue.

It could also happen that the paper's information needs to be updated, for example, the paper is accepted to a conference or journal. You can update the information by making a Pull Request and explaining the issue.
