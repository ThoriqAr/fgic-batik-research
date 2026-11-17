import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime

from matplotlib.backends.backend_pdf import (
    PdfPages
)


def export_pdf(
    visualizations,
    output_path=None
):

    if output_path is None:

        reports_dir = (
            Path.cwd() /
            "reports"
        )

        reports_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        timestamp = (
            datetime.now()
            .strftime("%Y%m%d_%H%M%S")
        )

        output_path = (

            reports_dir /

            f"qualitative_visualization_"
            f"{timestamp}.pdf"

        )

    else:

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    with PdfPages(
        output_path
    ) as pdf:

        for visualization in visualizations:

            figures = [

                visualization.original,

                visualization.explanation_a,

                visualization.explanation_b,

                visualization.explanation_c,

                visualization.explanation_d

            ]

            for figure in figures:

                pdf.savefig(
                    figure,
                    bbox_inches="tight"
                )

    print(
        "\nPDF saved:"
    )

    print(
        output_path
    )

    return output_path