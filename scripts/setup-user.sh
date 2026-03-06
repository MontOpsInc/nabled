#!/usr/bin/env bash
set -euo pipefail

NEW_USER="${NEW_USER:-agent}"
NEW_HOME="/home/${NEW_USER}"

if [[ "${EUID}" -eq 0 ]]; then
  SUDO=""
else
  SUDO="sudo"
fi

# 1) Create/normalize non-root user + sudo
if ! id -u "${NEW_USER}" >/dev/null 2>&1; then
  ${SUDO} adduser --disabled-password --gecos "" "${NEW_USER}"
fi
${SUDO} usermod -d "${NEW_HOME}" -s /bin/bash "${NEW_USER}"
${SUDO} usermod -aG sudo "${NEW_USER}"
${SUDO} install -d -m 700 -o "${NEW_USER}" -g "${NEW_USER}" "${NEW_HOME}/.ssh"

# 2) Seed authorized_keys from root if available (or keep existing key)
if [[ -f /root/.ssh/authorized_keys ]]; then
  ${SUDO} cp /root/.ssh/authorized_keys "${NEW_HOME}/.ssh/authorized_keys"
elif [[ ! -f "${NEW_HOME}/.ssh/authorized_keys" ]]; then
  echo "No authorized_keys found in /root/.ssh and none exists for ${NEW_USER}." >&2
  echo "Install ${NEW_HOME}/.ssh/authorized_keys before disabling root SSH." >&2
  exit 1
fi
${SUDO} chown "${NEW_USER}:${NEW_USER}" "${NEW_HOME}/.ssh/authorized_keys"
${SUDO} chmod 600 "${NEW_HOME}/.ssh/authorized_keys"

# 3) Passwordless sudo (automation convenience for ephemeral hosts)
${SUDO} tee "/etc/sudoers.d/90-${NEW_USER}" >/dev/null <<EOF
${NEW_USER} ALL=(ALL) NOPASSWD:ALL
EOF
${SUDO} chmod 440 "/etc/sudoers.d/90-${NEW_USER}"
${SUDO} visudo -cf "/etc/sudoers.d/90-${NEW_USER}"

# 4) Ensure HOME is stable in login shells
${SUDO} tee "${NEW_HOME}/.bash_profile" >/dev/null <<EOF
export HOME=${NEW_HOME}
EOF
${SUDO} chown "${NEW_USER}:${NEW_USER}" "${NEW_HOME}/.bash_profile"
${SUDO} tee /etc/profile.d/00-home-from-passwd.sh >/dev/null <<'EOF'
#!/usr/bin/env bash
set -eu
expected_home="$(getent passwd "$(id -u)" | awk -F: '{print $6}')"
if [[ -n "${expected_home:-}" && "${HOME:-}" != "${expected_home}" ]]; then
  export HOME="${expected_home}"
fi
EOF
${SUDO} chmod 644 /etc/profile.d/00-home-from-passwd.sh

# 5) Disable root SSH/password auth for remote access
${SUDO} install -d -m 755 /etc/ssh/sshd_config.d
${SUDO} tee /etc/ssh/sshd_config.d/99-agent-hardening.conf >/dev/null <<EOF
PermitRootLogin prohibit-password
PasswordAuthentication no
KbdInteractiveAuthentication no
ChallengeResponseAuthentication no
PubkeyAuthentication yes
AllowUsers ${NEW_USER} root
EOF

if command -v sshd >/dev/null 2>&1; then
  ${SUDO} sshd -t
fi

# 6) Reload SSH daemon where possible (container-safe fallbacks)
if command -v systemctl >/dev/null 2>&1 && [[ -d /run/systemd/system ]]; then
  ${SUDO} systemctl reload ssh || ${SUDO} systemctl reload sshd || true
elif command -v service >/dev/null 2>&1; then
  ${SUDO} service ssh reload || ${SUDO} service sshd reload || true
else
  ${SUDO} pkill -HUP sshd || true
fi

echo "setup complete for user ${NEW_USER}"
