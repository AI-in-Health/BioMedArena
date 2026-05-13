FROM rocker/r-ver:4.4.2

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    git \
    libbz2-dev \
    libcurl4-openssl-dev \
    libfontconfig1-dev \
    libfreetype6-dev \
    libfribidi-dev \
    libgit2-dev \
    libharfbuzz-dev \
    libjpeg-dev \
    liblzma-dev \
    libpng-dev \
    libssl-dev \
    libtiff5-dev \
    libuv1-dev \
    libxml2-dev \
    pkg-config \
    python3 \
    python3-pip \
    python3-venv \
    zlib1g-dev \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --break-system-packages --no-cache-dir \
    numpy \
    pandas \
    scipy \
    scikit-learn \
    statsmodels \
    matplotlib \
    seaborn \
    jupyter

RUN R -q -e "install.packages(c('BiocManager', 'remotes', 'tidyverse', 'data.table', 'MASS', 'jsonlite', 'readxl', 'ggnewscale'), repos='https://cloud.r-project.org', Ncpus=parallel::detectCores())"

RUN R -q -e "remotes::install_version('ggplot2', version='3.5.2', repos='https://cloud.r-project.org', upgrade='never')"

RUN R -q -e "BiocManager::install(c('DESeq2', 'apeglm', 'clusterProfiler', 'org.Hs.eg.db', 'enrichplot'), ask=FALSE, update=FALSE, Ncpus=parallel::detectCores()); stopifnot(requireNamespace('DESeq2', quietly=TRUE), requireNamespace('apeglm', quietly=TRUE), requireNamespace('clusterProfiler', quietly=TRUE), requireNamespace('org.Hs.eg.db', quietly=TRUE), requireNamespace('enrichplot', quietly=TRUE))"

WORKDIR /work

CMD ["bash"]
