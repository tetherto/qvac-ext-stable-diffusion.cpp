import test from 'node:test'
import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { join } from 'node:path'
import {
  REUSABLE_WORKFLOW,
  RUNNERS_YAML,
  assertReusableMatchesCatalog,
  findHardcodedLabelViolations,
  findMissingRunnerNamesNeeds,
  listAddonWorkflows,
  loadRunners,
  outputValue,
  parseRunnersYaml,
  readRepoFile,
  renderReusableWorkflow,
  repoRoot,
  runsOnExpression,
} from '../lib/runner-names.mjs'

test('runners.yaml parses with unique keys/targets', () => {
  const runners = loadRunners()
  assert.ok(runners.length >= 1)
  assert.equal(new Set(runners.map((e) => e.key)).size, runners.length)
  assert.equal(new Set(runners.map((e) => e.target ?? e.label)).size, runners.length)
})

test('the catalog still describes this repo: hosted images only', () => {
  // qvac-ext-stable-diffusion.cpp runs entirely on GitHub-hosted images. If a
  // self-hosted entry appears here, the job wiring needs the fork-ci authorize
  // gate that the self-hosted engine repos carry, and this repo has none.
  for (const entry of loadRunners()) {
    const labels = entry.kind === 'array' ? entry.labels : [entry.label]
    assert.ok(
      !labels.includes('self-hosted'),
      `${entry.key} is self-hosted; this repo has no authorize gate (see .github/RUNNERS.md)`,
    )
  }
})

test('array vs scalar output value + runs-on expression', () => {
  const [scalar] = parseRunnersYaml('windows_2022: windows-2022\n')
  assert.equal(outputValue(scalar), 'windows-2022')
  assert.equal(runsOnExpression(scalar), '${{ needs.runner_names.outputs.windows_2022 }}')

  const [arr] = parseRunnersYaml('gpu: [self-hosted, Linux, X64]\n')
  assert.equal(outputValue(arr), '["self-hosted","Linux","X64"]')
  assert.equal(runsOnExpression(arr), '${{ fromJSON(needs.runner_names.outputs.gpu) }}')
})

test('parseRunnersYaml rejects quoted values and junk', () => {
  assert.throws(() => parseRunnersYaml('k: "windows-2022"\n'), /invalid/)
  assert.throws(() => parseRunnersYaml('k: [self-hosted, "Linux"]\n'), /bare/)
  assert.throws(() => parseRunnersYaml('k: a\nk: b\n'), /duplicate runner key/)
  assert.throws(() => parseRunnersYaml('not yaml at all\n'), /invalid/)
})

test('reusable-runner-names.yml matches the catalog', () => {
  const runners = loadRunners()
  assert.doesNotThrow(() => assertReusableMatchesCatalog(runners, readRepoFile(REUSABLE_WORKFLOW)))
  const rendered = renderReusableWorkflow(runners)
  assert.match(rendered, /AUTO-GENERATED/)
  assert.match(rendered, /runs-on: ubuntu-latest/)
  assert.doesNotMatch(rendered, /actions\/checkout/)
})

test('CI workflows do not hardcode catalog runner targets', () => {
  const runners = loadRunners()
  const findings = []
  for (const file of listAddonWorkflows()) {
    const source = readRepoFile(file)
    findings.push(...findHardcodedLabelViolations(file, source, runners))
    findings.push(...findMissingRunnerNamesNeeds(file, source))
  }
  assert.deepEqual(findings, [], JSON.stringify(findings, null, 2))
})

test('the validated allowlist is not empty', () => {
  // A wired workflow absent from ADDON_WORKFLOWS is not checked at all, so an
  // empty or stale list makes every other assertion here vacuous.
  const wired = listAddonWorkflows()
  assert.ok(wired.length >= 1)
  assert.ok(wired.includes('.github/workflows/build.yml'))
})

test('detector catches scalar/array hardcoded runs-on but not matrix.os identities', () => {
  const runners = parseRunnersYaml(
    'windows_2022: windows-2022\nmacos_12: macos-12\nubuntu_2004: ubuntu-20.04\ngpu: [self-hosted, Linux, X64]\n',
  )
  const source = [
    'jobs:',
    '  a:',
    '    runs-on: windows-2022',
    '  b:',
    '    runs-on: [self-hosted, Linux, X64]',
    '  c:',
    "    runs-on: ${{ matrix.os == 'macos-12' && needs.runner_names.outputs.macos_12 || needs.runner_names.outputs.ubuntu_2004 }}",
    '  d:',
    '    runs-on: ${{ matrix.os }}',
    '  e:',
    '    runs-on: ubuntu-latest',
    '',
  ].join('\n')
  const findings = findHardcodedLabelViolations('x.yml', source, runners)
  assert.deepEqual(
    findings.map((f) => [f.line, f.target]).sort((a, b) => a[0] - b[0]),
    [
      [3, 'windows-2022'],
      [5, '[self-hosted,Linux,X64]'],
    ],
  )
})

test('detector flags a quoted catalog label used as a value in an expression', () => {
  const runners = parseRunnersYaml('windows_2022: windows-2022\n')
  const source = ['jobs:', '  a:', "    runs-on: ${{ inputs.win && 'windows-2022' || 'ubuntu-latest' }}", ''].join('\n')
  assert.equal(findHardcodedLabelViolations('x.yml', source, runners).length, 1)
})

test('validate-runner-names.mjs exits 0', () => {
  const result = spawnSync(process.execPath, [join(repoRoot, '.github/scripts/validate-runner-names.mjs')], {
    encoding: 'utf8',
    cwd: repoRoot,
  })
  assert.equal(result.status, 0, `stdout=${result.stdout}\nstderr=${result.stderr}`)
})

test('catalog path constant is correct', () => {
  assert.equal(RUNNERS_YAML, '.github/runners.yaml')
})
