import { Color3 } from '@babylonjs/core/Maths/math.color';
import { roleColor as hexRoleColor } from '@sliced/shared';

export { Role, roleLabel } from '@sliced/shared';
export type { SegbinData } from '@sliced/shared';

export function roleColor(r: number): Color3 {
  return Color3.FromHexString(hexRoleColor(r));
}
