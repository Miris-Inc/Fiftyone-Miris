import { executeOperator } from "@fiftyone/operators";

export async function executeOperatorAndReturn(
  uri: string,
  params?: Record<string, unknown>,
): Promise<Record<string, unknown>> {
  const result = await executeOperator(uri, params);
  return result?.result ?? {};
}
