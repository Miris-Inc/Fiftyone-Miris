/**
 * Type declarations for @fiftyone/* packages.
 *
 * These packages are not published to npm. They are provided at runtime
 * by the FiftyOne App. These stubs let TypeScript compile the plugin
 * during development without the full FiftyOne monorepo.
 *
 * Only the symbols this plugin actually imports are declared here.
 */

declare module "@fiftyone/operators" {
  export class OperatorConfig {
    constructor(options: {
      name: string;
      label: string;
      unlisted?: boolean;
      dynamic?: boolean;
    });
  }

  export interface ExecutionContext {
    params: Record<string, unknown>;
    dataset: unknown;
  }

  export function executeOperator(
    uri: string,
    params?: Record<string, unknown>,
  ): Promise<{ result?: Record<string, unknown> } | undefined>;

  export namespace types {
    class Object {
      str(name: string, options?: {
        label?: string;
        description?: string;
        placeholder?: string;
        required?: boolean;
        default?: string;
      }): void;
    }
    class Property {
      constructor(type: types.Object);
    }
  }

  export abstract class Operator {
    abstract get config(): OperatorConfig;
    resolveInput(ctx: ExecutionContext): types.Property | void;
    execute(ctx: ExecutionContext): Promise<void> | void;
  }

  export function registerOperator(
    operator: typeof Operator,
    pluginName: string,
  ): void;
}

