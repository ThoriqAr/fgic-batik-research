import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages


def export_pdf(
    results,
    attribution_dir,
    report_type="per_class"
):

    attribution_dir = Path(
        attribution_dir
    )

    experiment_dir = attribution_dir.parent

    reports_dir = (
        experiment_dir /
        "reports"
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_pdf = (
        reports_dir /
        (
            f"{report_type}"
            f"_visualization_report_"
            f"{timestamp}.pdf"
        )
    )

    with PdfPages(output_pdf) as pdf:

        for visualization in results:

            pdf.savefig(
                visualization.figure,
                bbox_inches="tight"
            )

            plt.close(visualization.figure)

    print(
        f"PDF saved:\n{output_pdf}"
    )