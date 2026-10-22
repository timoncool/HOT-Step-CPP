import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import tseslint from 'typescript-eslint'
import { defineConfig, globalIgnores } from 'eslint/config'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      js.configs.recommended,
      tseslint.configs.recommended,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
    },
    rules: {
      // fetchQuery is deprecated in TanStack Query v5 (removed next major);
      // queryClient.query() has identical semantics. Keep the P1 sweep permanent.
      'no-restricted-syntax': [
        'error',
        {
          selector: "CallExpression[callee.property.name='fetchQuery']",
          message: 'queryClient.fetchQuery is deprecated — use queryClient.query() (docs/dev/tanstack-query.md).',
        },
      ],
    },
  },
])
