/**
 * Type declarations for @fiftyone/* packages and runtime-provided globals.
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
      execute_as_generator?: boolean;
    });
  }

  export interface ExecutionContext {
    params: Record<string, unknown>;
    dataset: { name: string } | null;
    hooks: Record<string, unknown>;
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
      int(name: string, options?: {
        label?: string;
        description?: string;
        required?: boolean;
        default?: number;
        min?: number;
        max?: number;
      }): void;
      enum(name: string, values: string[], options?: {
        label?: string;
        description?: string;
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
    useHooks?(): Record<string, unknown>;
    resolveInput(ctx: ExecutionContext): types.Property | void;
    execute(ctx: ExecutionContext): Promise<void> | void;
  }

  export function registerOperator(
    operator: typeof Operator,
    pluginName: string,
  ): void;
}

declare module "@fiftyone/state" {
  export interface ModalSample {
    sample: Record<string, unknown> | null;
  }
  export const modalSample: unknown;
  export const datasetName: unknown;
}

declare module "recoil" {
  export function useRecoilValue<T>(state: unknown): T;
}
