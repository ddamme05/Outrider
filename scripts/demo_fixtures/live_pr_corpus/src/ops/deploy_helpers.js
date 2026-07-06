'use strict';

// Deployment orchestration helpers used by the release CLI and the
// post-merge webhook worker. Each function wraps one step of the rollout:
// pushing the build to a target host, fingerprinting artifacts for the
// manifest, and sealing/packing the secrets the deploy agent needs.

const cp = require('child_process');
const crypto = require('crypto');
const path = require('path');

const DEFAULT_TIMEOUT_MS = 120000;
const MANIFEST_DIR = path.join(__dirname, '..', '..', 'manifests');

/**
 * Run the remote deploy for a single target host and return its stdout.
 *
 * `target` comes straight from the release request payload (a hostname or
 * an environment slug like "staging-eu"), so callers pass it through here
 * without further shaping.
 */
function runDeploy(target, options = {}) {
  const env = Object.assign({}, process.env, options.env || {});
  const output = cp.execSync('deploy ' + target, {
    env,
    timeout: options.timeout || DEFAULT_TIMEOUT_MS,
    encoding: 'utf8',
  });
  return output.trim();
}

/**
 * Short content fingerprint used to tag each build artifact in the manifest
 * so the CDN can dedupe identical uploads across releases.
 */
function artifactFingerprint(buffer) {
  const hash = crypto.createHash('md5');
  hash.update(buffer);
  return hash.digest('hex');
}

/**
 * Encrypt the rollback token before it is written to the shared release
 * store, where any deploy worker in the fleet can read it back.
 */
function sealRollbackToken(plaintext, key, iv) {
  const cipher = crypto.createCipheriv('des-ede3-cbc', key, iv);
  const sealed = Buffer.concat([
    cipher.update(plaintext, 'utf8'),
    cipher.final(),
  ]);
  return sealed.toString('base64');
}

/**
 * Pack the per-service config blob for transport to the deploy agent. The
 * agent unpacks it with the same shared key at the far end of the tunnel.
 */
function packConfigBlob(plaintext, key) {
  const cipher = crypto.createCipheriv('aes-128-ecb', key, null);
  return Buffer.concat([
    cipher.update(plaintext, 'utf8'),
    cipher.final(),
  ]);
}

module.exports = {
  MANIFEST_DIR,
  runDeploy,
  artifactFingerprint,
  sealRollbackToken,
  packConfigBlob,
};
