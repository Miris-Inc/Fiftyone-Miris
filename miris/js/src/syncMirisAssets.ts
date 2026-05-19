import { Operator, OperatorConfig, ExecutionContext, executeOperator, types } from "@fiftyone/operators";
import { getMirisScene } from "./mirisScene";

const PLUGIN_NAME = "@miris-inc/voxel51";
const UPSERT_OP = `${PLUGIN_NAME}/upsert_miris_asset`;
const DEFAULT_VIEWER_KEY = "4YIGMPUj5-fL8n0jkp1kQpJktss_UaBDMW9jwJb08f4";

async function fetchMirisAssets(viewerKey: string) {
  const scene = await getMirisScene();
  scene.viewerKey = viewerKey;
  return scene.fetchAssets();
}

class SyncMirisAssets extends Operator {
  get config(): OperatorConfig {
    return new OperatorConfig({
      name: "sync_miris_assets",
      label: "Sync Miris Assets",
    });
  }

  resolveInput(_ctx: ExecutionContext): types.Property {
    const inputs = new types.Object();
    inputs.str("viewer_key", {
      label: "Viewer Key",
      placeholder: DEFAULT_VIEWER_KEY,
      required: false,
    });
    return new types.Property(inputs);
  }

  async execute(ctx: ExecutionContext): Promise<void> {
    const viewerKey = (ctx.params.viewer_key as string) || DEFAULT_VIEWER_KEY;
    const assets = await fetchMirisAssets(viewerKey);

    for (const asset of assets) {
      try {
        await executeOperator(UPSERT_OP, {
          uuid: asset.uuid,
          name: asset.name,
          thumbnail: asset.thumbnailUrl,
          viewer_key: viewerKey,
          tags: asset.tags,
        });
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : String(err);
        if (message.includes("No dataset is currently loaded")) {
          throw new Error(
            "No dataset is currently loaded. Open a dataset in FiftyOne before syncing Miris assets."
          );
        }
        throw err;
      }
    }

    await executeOperator("@voxel51/operators/reload_dataset", {});
  }
}

export { SyncMirisAssets, DEFAULT_VIEWER_KEY };
