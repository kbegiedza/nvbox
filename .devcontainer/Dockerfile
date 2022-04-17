FROM ghcr.io/kbegiedza/cuda:11.6.0-sdk

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ARG INSTALL_ZSH="false"
ARG UPGRADE_PACKAGES="false"
ARG COMMON_SCRIPT_SOURCE="https://raw.githubusercontent.com/microsoft/vscode-dev-containers/main/script-library/common-debian.sh"
ARG COMMON_SCRIPT_SHA="dev-mode"

RUN apt-get update \
    && ([ "${COMMON_SCRIPT_SHA}" = "dev-mode" ] || (echo "${COMMON_SCRIPT_SHA} */tmp/common-setup.sh" | sha256sum -c -)) \
    && /bin/bash /tmp/common-setup.sh "${INSTALL_ZSH}" "${USERNAME}" "${USER_UID}" "${USER_GID}" "${UPGRADE_PACKAGES}" \
    ## Post installation clean up
    && rm /tmp/common-setup.sh \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*